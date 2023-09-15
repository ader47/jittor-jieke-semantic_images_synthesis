"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import jittor
from models.guided_diffusion.image_datasets import load_data
from models.guided_diffusion import logger
from models.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from models.autoencoder import instantiate_from_config
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from models.guided_diffusion.script_util import preprocess_input, Normalize

import os, random, json, argparse


def custom_to_pil(x):
    x = (x.detach() + 1.) / 2.
    x = (255 * x).clamp(0, 255).permute(1, 2, 0)
    x = Image.fromarray(x.data.astype(np.uint8))
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def setup_seed(seed):
    jittor.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(kwargs):
    config = create_argparser()
    config.update(kwargs)
    logger.configure()
    logger.log('sampling config')
    logger.log(config)
    os.makedirs(config['output_path'], exist_ok=True)
    logger.log("creating model and diffusion ...")
    model, diffusion = create_model_and_diffusion(
        **{k: config[k] for k in model_and_diffusion_defaults().keys()}
    )
    model.eval()
    model.load_state_dict(jittor.load(config['model_path']))
    model.cuda()

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=config['dataset_mode'],
        data_dir=config['input_path'],
        batch_size=config['batch_size'] * jittor.world_size,
        image_size=config['image_size'],
        class_cond=config['class_cond'],
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False,
        is_feature=True,
    )

    logger.log("loading AutoEncoder ...")
    ae = instantiate_from_config(OmegaConf.load(config['ae_config']).model)
    ae.load_state_dict(jittor.load(config['ae_ckpt']))
    ae.cuda()

    file_name = json.load(
        open(os.path.join(config['input_path'], 'label_to_img.json'), 'r'))

    if jittor.rank == 0:
        samples_count = 0

    logger.log("start sampling...")
    for i, (labels, label_dic) in enumerate(data):
        model_kwargs = preprocess_input(label_dic, num_classes=config['num_classes'])
        model_kwargs['s'] = config["s"]
        sample_fn = (
            diffusion.p_sample_loop if not config['use_ddim'] else diffusion.ddim_sample_loop
        )

        x_0 = []
        for j in range(labels.shape[0]):
            samplename = file_name[label_dic['file_name'][j]]
            if config['resume_image'] != '':
                img = Image.open(os.path.join(config['resume_image'], label_dic['file_name'][j].split('.')[0] + '.jpg'))
            else:
                img = Image.open(os.path.join(config['img_path'], 'imgs', samplename))
            image = np.array(img).astype(np.float32)
            image = (image / 127.5 - 1.0).astype(np.float32)
            img = jittor.array(image).permute(2, 0, 1)
            x_0.append(img)
        x_0 = jittor.stack(x_0)

        with jittor.no_grad():
            x_0 = ae.encode(x_0)

        t = [config['start_t'] for i in range(x_0.shape[0])]
        indices = jittor.array(np.array(t)).long()
        x_t = diffusion.q_sample(x_0, t=indices)

        sample = sample_fn(
            model,
            (config['batch_size'], 3, config['image_size'][1], config['image_size'][0]),
            noise=x_t,
            clip_denoised=config['clip_denoised'],
            model_kwargs=model_kwargs,
            progress=True,
            t_start=config['start_t']
        )

        with jittor.no_grad():
            z = ae.decode(sample.detach())

        for j in range(z.shape[0]):
            img = custom_to_pil(z[j])
            img.save(os.path.join(config['output_path'], label_dic['file_name'][j].split('.')[0] + '.jpg'))

        if jittor.rank == 0:
            samples_count += (sample.shape[0] * jittor.world_size)
            logger.log(f"\n created {samples_count} samples")
            if samples_count > config['num_samples']:
                break


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=False,
        num_samples=10000,
        batch_size=4,
        use_ddim=True,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0
    )
    defaults.update(model_and_diffusion_defaults())

    # model config
    defaults['attention_resolutions'] = '32, 16, 8'
    defaults['diffusion_steps'] = 1000
    defaults['learn_sigma'] = True
    defaults['noise_schedule'] = 'linear'
    defaults['num_channels'] = 256
    defaults['num_head_channels'] = 64
    defaults['num_res_blocks'] = 2
    defaults['resblock_updown'] = True
    defaults['use_scale_shift_norm'] = True
    defaults['model_path'] = 'checkpoints/average-176k-266k.pkl'
    defaults['use_fp16'] = False
    defaults['num_classes'] = 29
    defaults['class_cond'] = True
    defaults['ae_config'] = 'config/config-jittor.yaml'
    defaults['ae_ckpt'] = './checkpoints/autoencoder.pkl'

    # data config
    defaults['dataset_mode'] = 'jittor'
    defaults['no_instance'] = True

    # sampling config
    defaults['is_train'] = False
    defaults['image_size'] = (128, 96)
    defaults['use_ddim'] = True

    return defaults


if __name__ == "__main__":
    setup_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--output_path', type=str, default='./result')
    parser.add_argument('-s', type=float, default=3.5)
    parser.add_argument('--start_t', type=int, default=300)
    parser.add_argument('--timestep_respacing', type=str, default='ddim500')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--resume_image', type=str, default='')
    parser.add_argument('--num_samples',type=int,default=1000)
    args = parser.parse_args()

    if args.resume_image != '':
        os.makedirs(args.resume_image+'-colored', exist_ok=True)
        jsons = json.load(
            open(os.path.join(args.input_path, 'label_to_img.json'), 'r'))
        file_name = os.listdir(args.resume_image)
        for i in file_name:
            source = i.split('.')[0] + '.jpg'
            ref = jsons[i.split('.')[0] + '.png']
            source = Image.open(os.path.join(args.resume_image, source))
            ref = Image.open(os.path.join(args.img_path, 'imgs', ref))
            img = Normalize(ref, source)
            img.save(os.path.join(args.resume_image+'-colored', i.split('.')[0] + '.jpg'))
        args.resume_image = args.resume_image+'-colored'
    main(args.__dict__)
