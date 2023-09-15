import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()
    os.environ['OPENAI_LOGDIR'] = './logs/sample'
    # first stage
    os.system('python image_sample.py' +
              ' --input_path ' + args.input_path +
              ' --img_path ' + args.img_path +
              ' --output_path ./temp/stage0' +
              ' -s 3.5 --timestep_respacing ddim500 --start_t 300 --batch_size 5 --num_samples 1000')
    # second stage
    os.system('python image_sample.py' +
              ' --input_path ' + args.input_path +
              ' --img_path ' + args.img_path +
              ' --output_path ./temp/stage1' +
              ' --resume_image ' + './temp/stage0' +
              ' -s 3.7 --timestep_respacing ddim500 --start_t 320 --batch_size 5 --num_samples 1000')
    # last stage
    os.system('python image_sample.py' +
              ' --input_path ' + args.input_path +
              ' --output_path ' + args.output_path +
              ' --img_path ' + args.img_path +
              ' --resume_image ' + './temp/stage1' +
              ' -s 3.7 --timestep_respacing ddim50 --start_t 20 --batch_size 5 --num_samples 1000')
    # copy selected images to ./selects
    os.makedirs('selects', exist_ok=True)
    with open('./select.txt','r') as f:
        for line in f:
            file_name = line.strip('\n')
            os.system('cp '+os.path.join(args.output_path,file_name)+' ./selects')
    # del temp images
    os.system('rm -r temp')
