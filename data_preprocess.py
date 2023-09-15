from omegaconf import OmegaConf
from models.autoencoder.util import instantiate_from_config
import jittor
import os,argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from npy_append_array import NpyAppendArray

def preprocess(input_path, output_path):
    '''
    :param input_path: dataset path
    :param output_path: save path
    :return: None
    '''
    ae = instantiate_from_config(OmegaConf.load('config/config-jittor.yaml').model)
    ae.load_state_dict(
        jittor.load('checkpoints/autoencoder.pkl'))
    ae.cuda()
    # prepare data
    image_path = os.path.join(input_path, 'imgs')
    label_path = os.path.join(input_path, 'labels')
    fileName = os.listdir(image_path)
    print('image count:', len(fileName))
    # save filename
    path_feature = os.path.join(output_path, 'images.npy')
    path_labels = os.path.join(output_path, 'labels.npy')
    save_name = os.path.join(output_path, 'filename.txt')
    features = []
    labels = []
    for t in tqdm(range(len(fileName))):
        name = fileName[t]
        with open(save_name, 'a+') as f:
            f.writelines(name + '\n')
        image = Image.open(os.path.join(image_path, name))
        img_filp = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        img = jittor.array(image).unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        img_filp = np.array(img_filp).astype(np.uint8)
        img_filp = (img_filp / 127.5 - 1.0).astype(np.float32)
        img_filp = jittor.array(img_filp).unsqueeze(0)
        img_filp = img_filp.permute(0, 3, 1, 2)
        input = jittor.concat((img, img_filp), dim=0)
        with jittor.no_grad():
            z = ae.encode(input)
        features.append(z.detach().cpu())
        label = Image.open(os.path.join(label_path, name.split('.')[0] + '.png'))
        label = label.resize((128, 96), Image.NEAREST)
        label_filp = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        label = np.array(label)
        label_filp = np.array(label_filp)
        assert (label.min() >= 0), 'error'
        labels.append(np.expand_dims(label, 0))
        labels.append(np.expand_dims(label_filp, 0))
        if len(labels) % 1000 == 0:
            with NpyAppendArray(path_feature) as npaa:
                npaa.append(np.vstack(features))
                features.clear()
            with NpyAppendArray(path_labels) as npaa:
                npaa.append(np.vstack(labels))
                labels.clear()
    if len(features) > 0:
        with NpyAppendArray(path_feature) as npaa:
            npaa.append(np.vstack(features))
            features.clear()
        with NpyAppendArray(path_labels) as npaa:
            npaa.append(np.vstack(labels))
            labels.clear()


if __name__ == '__main__':
    print('start preprocess')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    kwargs = parser.parse_args()
    preprocess(**kwargs.__dict__)
    print('preprocess finish!')
