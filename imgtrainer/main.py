from img_train import ImgTrainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_path', type=str, default='/root/autodl-tmp/TTMI/imgtrainer/config.yaml')
    args = parser.parse_args()       

    mmtrainer = ImgTrainer(args.config_path)
    print('start training')
    mmtrainer.train()