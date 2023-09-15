"""
Train a diffusion model on latent space.
"""
import argparse

import jittor

from models.guided_diffusion import logger
from models.guided_diffusion import load_data

from models.guided_diffusion.resample import create_named_schedule_sampler

from models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from models.guided_diffusion.train_util import TrainLoop


def main(kwargs):
    config = create_argparser()
    config.update(kwargs)
    logger.configure()
    logger.log("current total batch is ", config['batch_size'])
    logger.log("creating model and diffusion...")
    # todo
    model, diffusion = create_model_and_diffusion(
        **{k: config[k] for k in model_and_diffusion_defaults().keys()}
    )
    model.cuda()
    model.train()

    if config['use_fp16']:
        jittor.flags.auto_mixed_precision_level = 6
        model.float_auto()
    schedule_sampler = create_named_schedule_sampler(config['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=config['dataset_mode'],
        data_dir=config['input_path'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        class_cond=config['class_cond'],
        is_train=config['is_train'],
    )
    logger.log("start training...")

    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=config['num_classes'],
        batch_size=config['batch_size'],
        microbatch=config['microbatch'],
        lr=config['lr'],
        ema_rate=config['ema_rate'],
        drop_rate=config['drop_rate'],
        log_interval=config['log_interval'],
        save_interval=config['save_interval'],
        resume_checkpoint=config['resume_checkpoint'],
        use_fp16=config['use_fp16'],
        fp16_scale_growth=config['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=config['weight_decay'],
        lr_anneal_steps=config['lr_anneal_steps'],
        latent_scale=config['latent_scale']
    )

    trainer.run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables micro batches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True,
        latent_scale=False
    )
    defaults.update(model_and_diffusion_defaults())
    defaults['image_size'] = (128, 96)
    defaults['num_channels'] = 256
    defaults['num_head_channels'] = 64
    defaults['lr'] = 1e-4
    defaults['attention_resolutions'] = "32,16,8"
    defaults['diffusion_steps'] = 1000
    defaults['learn_sigma'] = True
    defaults['noise_schedule'] = "linear"
    defaults['num_res_blocks'] = 2
    defaults['resblock_updown'] = True
    defaults['use_scale_shift_norm'] = True
    defaults['use_checkpoint'] = True
    defaults['num_classes'] = 29
    defaults['class_cond'] = True
    defaults['no_instance'] = True
    defaults['drop_rate'] = 0.2
    defaults['use_fp16'] = False
    defaults['log_interval'] = 10
    defaults['save_interval'] = 10000
    defaults['microbatch'] = 5
    defaults['batch_size'] = defaults['microbatch'] * jittor.world_size
    defaults['dataset_mode'] = 'jittor'

    return defaults


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    args = parser.parse_args()

    main(args.__dict__)


