import copy
import functools
import os

import blobfile as bf
from datetime import datetime
import jittor
from jittor.optim import AdamW

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import UniformSampler
from mpi4py import MPI

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            num_classes,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            drop_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            latent_scale=False
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.scale_factor = 1
        self.latent_scale = latent_scale

        self.step = 0
        self.resume_step = 0

        self.global_batch = self.batch_size  # jittor 是把batch 拆分给各张卡的 # * jittor.world_size
        # mpi for boardcasting
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self._load_and_sync_parameters()
        # todo Trainer
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
            if jittor.world_size > 1:
                self.ema_params = self.comm.bcast(self.ema_params, 0)
                print("ema weight in rank", jittor.rank, ", check is it same? ")
                print(self.ema_params[0][-1])
        else:
            # use numpy, avoid GPU memory used.
            self.ema_params = []
            # 必须拷贝，否则共享内存，ema会出错！！
            temp = copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate)):
                res = []
                for i in temp:
                    res.append(i.detach().clone().cpu().data)
                self.ema_params.append(res)
            print(self.mp_trainer.master_params[-1])
            print(self.ema_params[0][-1])

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if jittor.rank == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    jittor.load(
                        resume_checkpoint
                    )
                )
        # 第一个进程加载完成ckpt之后，将参数传播给后续进程
        # 将一个张量从指定的进程广播到所有其他进程。
        # https://blog.csdn.net/BlackZhou013/article/details/129676975
        # todo 单卡训练不需要这个
        # dist_util.sync_params(self.model.parameters())
        if jittor.world_size > 1:
            self.model.mpi_param_broadcast(0)
        # for i in range(jittor.world_size):
        #     print(self.model.parameters()[-1])

    def _load_ema_parameters(self, rate):
        res = []
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            # state_dict = jittor.load(
            #     ema_checkpoint
            # )
            # res = [state_dict[key] for key in self.model.state_dict().keys()]
            if self.rank == 0:
                state_dict = jittor.load(
                    ema_checkpoint
                )
                res = [state_dict[key] for key in self.model.state_dict().keys()]
            else:
                state_dict = None
            # state_dict = self.comm.bcast(state_dict, root=0)
        return res

    def _load_optimizer_state(self):
        # cannot broadcast, overflow
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pkl"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = jittor.load(
                opt_checkpoint
            )
            if state_dict['defaults']['lr'] != self.lr:
                state_dict['defaults']['lr'] = self.lr
            print("Current learning rate is setting to", state_dict['defaults']['lr'])
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)  # 取出数据
            # print(batch.shape)
            # vq-f4 不需要进行scale
            if self.latent_scale:
                if self.model.scale_factory is None:
                    self.model.scale_factory = 1. / batch.flatten().std()
                    logger.log("scale factory set to {}".format(self.model.scale_factory))
                # 保存的是encode之后的数据, 需要scale
                batch = batch * self.model.scale_factory
            cond = self.preprocess_input(cond)
            self.run_step(batch, cond)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if jittor.rank == 0:
                self.step += 1
            # only main thread add step
            if self.step % self.save_interval == 0:
                self.save()
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        # for i in self.mp_trainer.master_params:
        #     print(i.opt_grad(self.opt))
        # todo fp16 loss Nan inf
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        # gradient accumulate
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # print(i)
            logger.logkv("micro batch", self.microbatch)
            micro = batch[i: i + self.microbatch]
            micro_cond = {
                k: v[i: i + self.microbatch]
                for k, v in cond.items()
            }
            # todo last batch if batch can be divided by microbatch last batch is not used
            # must make sure batch can be divided by microbatch
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            # todo sample t and weights
            t, weights = self.schedule_sampler.sample(micro.shape[0])

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            losses = compute_losses()
            # if isinstance(self.schedule_sampler, LossAwareSampler):
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses["loss"].detach()
            #     )
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss, self.opt)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if jittor.rank == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pkl"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pkl"
                jittor.save(state_dict, os.path.join(get_blob_logdir(), filename))

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if jittor.rank == 0:
            logger.log(f"saving opt...")
            jittor.save(self.opt.state_dict(),
                        os.path.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pkl"))
        logger.log("save ckpt finish")
        # todo what's this?
        # dist.barrier()

    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        input_label = jittor.init.zero_(jittor.randn((bs, nc, h, w)))
        src = jittor.init.one_(jittor.randn((bs, nc, h, w)))
        input_semantics = input_label.scatter_(dim=1, index=label_map, src=src)
        if 'instance' in data:
            # todo instance
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = jittor.concat((input_semantics, instance_edge_map), dim=1)

        if self.drop_rate > 0.0:
            mask = (jittor.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate).float()
            input_semantics = input_semantics * mask

        cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'path', 'label_ori']}
        cond['y'] = input_semantics
        return cond

    def get_edges(self, t):
        edge = jittor.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pkl, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pkl"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        # temp = values.mean().item()
        temp = values.mean()
        logger.logkv_mean(key, temp)
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().data, values.detach().cpu().data):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
