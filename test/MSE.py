import argparse
import glob
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    parser.add_argument('--mask_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    parser.add_argument('--real_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    args = parser.parse_args()
    
    pred_path = args.pred_path
    real_path = args.real_path
    mask_path = args.mask_path
    
    pred_img_lists = sorted(glob.glob(os.path.join(pred_path,"*.png")))
    real_img_lists = sorted(glob.glob(os.path.join(real_path,"*.png")))
    mask_img_lists = sorted(glob.glob(os.path.join(mask_path,"*.png")))
    
    print(len(pred_img_lists),len(real_img_lists))
    assert len(pred_img_lists)==len(real_img_lists)
    
    mse_rgb = 0
    mse_g = 0
    
    for i,img in enumerate(pred_img_lists):
        pred = cv2.imread(img)
        gray_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        
        gt = cv2.imread(real_img_lists[i])
        gt = cv2.resize(gt,(256,256))
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.resize(gt_gray,(256,256))
        
        mask = cv2.imread(mask_img_lists[i])
        mask = cv2.resize(mask,(256,256))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.resize(mask_gray,(256,256))
        mask_gray[mask_gray!=0] = 1
        mask_gray[mask_gray==0] = 255
        mask_gray[mask_gray==1] = 0
        num_nonzero = np.count_nonzero(mask_gray, axis=0)
        count = 0
        for l in num_nonzero:
            count += l
        assert count!=0
            
        
        # print("nonzeros ",count)
        
        mse = np.sum((gt-pred)**2)/count
        # print(mask_gray.shape)
        
        mse_gray = np.sum((gt_gray-gray_pred)**2)/count
        
    mse_rgb += mse
    mse_g += mse_gray
        
    print(mse_rgb/(i+1))
    print(mse_g/(i+1))

if __name__=='__main__':
    main()