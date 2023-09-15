"""
Various utilities for neural networks.
"""
import math
import jittor
import jittor.nn as nn


class SiLU(nn.Module):
    def execute(self, x):
        return x * jittor.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def execute(self, x):
        return super().execute(x.float()).astype(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        # todo jittor has not Avgpool1d
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for i in range(len(target_params)):
        # use numpy avoid out of gpu mem
        target_params[i] = target_params[i] * rate + (source_params[i].data) * (1 - rate)



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dims=tuple(range(1, len(tensor.shape)))) # not list must be tuple


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Var of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Var of positional embeddings.
    """
    half = dim // 2
    freqs = jittor.exp(
        -math.log(max_period) * jittor.arange(start=0, end=half, dtype=jittor.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = jittor.concat([jittor.cos(args), jittor.sin(args)], dim=-1)
    if dim % 2:
        embedding = jittor.concat([embedding, jittor.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs)+tuple(params)
        return CheckpointFunction.apply(func, params,len(inputs), *args)
    else:
        return func(*inputs)


# Gradient checkpoint
# https://zhuanlan.zhihu.com/p/455541708
class CheckpointFunction(jittor.Function):

    def execute(self, run_function, params,length, *args):
        self.run_function = run_function
        self.input_tensors = list(args[:length])
        self.params = params
        with jittor.no_grad():
            output_tensors = self.run_function(*self.input_tensors)

        return output_tensors

    def grad(self, *output_grads):
        self.input_tensors = [x.detach().start_grad() for x in self.input_tensors]
        with jittor.enable_grad():
            shallow_copies = [x.view_as(x) for x in self.input_tensors]
            output_tensors = self.run_function(*shallow_copies)
        output_tensors.mul_(output_grads[0].detach().clone().stop_grad())
        input_grads = jittor.grad(
            output_tensors,
            self.input_tensors + self.params,
            retain_graph=False
        )
        del self.input_tensors
        # del self.input_params
        del self.params
        del output_grads
        del output_tensors
        return (None,None,None)+tuple(input_grads)