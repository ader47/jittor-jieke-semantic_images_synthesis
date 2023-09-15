import argparse
import os

if __name__ == "__main__":
    # python train.py --input_path 训练数据路径，即可展开训练。
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    args = parser.parse_args()
    # data pre-process
    save_dataset = './datasets'
    os.makedirs(save_dataset, exist_ok=True)
    os.system('python data_preprocess.py'+' --input_path '+args.input_path+' --output_path '+save_dataset)
    # start train
    NUM_GPUs = 1  # set gpu num, bigger is better. recommend 8
    os.environ['OPENAI_LOGDIR'] = './logs/train'
    os.system('mpirun -np ' + str(NUM_GPUs) + ' --oversubscribe python image_train.py --input_path ' + save_dataset)
