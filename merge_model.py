import pickle
import numpy as np
import argparse

def merge_ckpt(save_path, ckpts_to_merge):
    model_weights = []
    for model_path in ckpts_to_merge:
        print(model_path)
        model_weights.append(pickle.load(open(model_path, 'rb+')))

    weight_avg = {}
    for key in model_weights[0].keys():
        ws = np.zeros(model_weights[0][key].shape)
        for model_weight in model_weights:
            ws += model_weight[key]
        weight_avg[key] = ws / len(model_weights)

    with open(save_path, 'wb') as fo:
        pickle.dump(weight_avg, fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str,default='./checkpoints/average-176-266.pkl')
    parser.add_argument('--ckpts',type=list,default=['./checkpoints/176000.pkl', './checkpoints/266000.pkl'])
    args = parser.parse_args()
    merge_ckpt(args.save_path, args.ckpts)
