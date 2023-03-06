import argparse
import glob
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default="")
    parser.add_argument('--real_path', default="")
    args = parser.parse_args()
    
    pred_path = args.pred_path
    real_path = args.real_path
    
    pred_img_lists = sorted(glob.glob(os.path.join(pred_path,"*.png")))
    real_img_lists = sorted(glob.glob(os.path.join(real_path,"*.png")))
    
    # l2 = np.sum(np.power((actual_value-predicted_value),2))
    # print(l2)
    
    assert len(pred_img_lists)==len(real_img_lists)
    
    l2_rgb = 0
    l2_g = 0
    
    for i,img in enumerate(pred_img_lists):
        pred = cv2.imread(img)
        gray_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        
        gt = cv2.imread(real_img_lists[i])
        gt = cv2.resize(gt,(256,256))
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.resize(gt_gray,(256,256))
        
        # l2 = np.sum(np.power((gt-pred),2))
        l2 = np.sum((gt-pred)**2)
        # print(l2,l2.shape)
        
        # l2_gray = np.sum(np.power((gt_gray-gray_pred),2))
        l2_gray = np.sum((gt_gray-gray_pred)**2)
        # print(l2_gray,l2.shape)
        l2_rgb += l2
        l2_g += l2_gray
        
    print(l2/(i+1))
    print(l2_g/(i+1))

if __name__=='__main__':
    main()

        
        
        
