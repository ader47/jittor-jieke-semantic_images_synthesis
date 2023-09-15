"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import jittor.nn as nn
import jittor

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight = l.weight.half()
        if l.bias is not None:
            l.bias= l.bias.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """

    master_params = []
    for param_group, shape in param_groups_and_shapes:
        a = [param.detach().float() for (_, param) in param_group]
        temp = jittor.concat(a).view(shape)   # 维度不同
        master_param = nn.Parameter(temp)
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            # print("---------------------------------------------------")
            # print(state_dict[name].shape)
            # print(master_params[i].shape)
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16,numpy=True):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        if numpy:
            master_params = [state_dict[name].data for name, _ in model.named_parameters()]
        else:
            master_params = [state_dict[name] for name, _ in model.named_parameters()]

    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return jittor.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        # 虽然地址不相同，但是值也是会同步变化的。
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale
        if self.use_fp16:
            # param_groups_and_shapes  parameters splited into vector group and matrix group
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )


    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss,opt):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            # loss_scale=1.0 / (2 ** self.lg_loss_scale)
            opt.backward(loss * loss_scale)
        else:
            opt.backward(loss)
    def optimize(self, opt: jittor.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: jittor.optim.Optimizer):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        grad_norm, param_norm = self._compute_norms(opt,grad_scale=2 ** self.lg_loss_scale)

        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            opt.zero_grad()
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].opt_grad(opt).mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        opt.zero_grad()
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: jittor.optim.Optimizer):
        opt.step()
        return True

    def _compute_norms(self,opt, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        if self.use_fp16:
            for param_group, shape in self.param_groups_and_shapes:
                with jittor.no_grad():
                    a=[param.view(-1).detach().float() for (_, param) in param_group]
                    p = jittor.concat(a).view(-1)
                    param_norm += float(jittor.norm(p, p=2).astype(jittor.float32) ** 2)

                    grad_temp = [param.opt_grad(opt).view(-1) for (_, param) in param_group]
                    p = jittor.concat(grad_temp).view(-1)
                    grad_norm += jittor.norm(p, p=2).astype(jittor.float32) ** 2
            return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
