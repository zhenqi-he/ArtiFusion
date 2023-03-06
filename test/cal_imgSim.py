from image_similarity_measures.evaluate import evaluation
import argparse
import glob
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    parser.add_argument('--real_path', default="/root/autodl-tmp/histology/predicted/aa_cyclegan/test_latest/images")
    args = parser.parse_args()
    
    pred_path = args.pred_path
    real_path = args.real_path
    
    pred_lists = sorted(glob.glob(os.path.join(pred_path,"*.png")))
    gt_lists = sorted(glob.glob(os.path.join(real_path,"*.png")))
    print("len(gt_lists)",len(gt_lists))
    print(len(gt_lists),len(pred_lists))
    assert len(gt_lists) == len(pred_lists)
    
    
    # evaluation(org_img_path=gt_lists[0], 
    #        pred_img_path=pred_lists[0], 
    #        metrics=["ssim", "psnr","fsim","issm","sre","sam","uiq"])
    
    # ssim = 0
    # psnr = 0
    # fsim = 0
    # issm = 0
    # sre = 0
    # sam = 0
    # uiq = 0
    metrics=["ssim", "psnr","fsim","issm","sre","sam","uiq"]
    d = {}
    for m in metrics:
        d[m] = 0
    for i,imgP in enumerate(gt_lists):
        print(i)
        e = evaluation(org_img_path=imgP, 
           pred_img_path=pred_lists[i], 
           metrics=metrics)
        for m in metrics:
            d[m] += e[m]
    
    for m in metrics:
        print("{} : {}".format(m,d[m]/(i+1)))
        
if __name__ == '__main__':
    main()

    
    
        