# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:59:22 2018

@author: Negar
"""


import cv2,os,time
import numpy as np
import math 
from detect import crop_row_detect
import time

### Setup ###
image_data_path = os.path.abspath('../CRBD/Images')
image_out_path = os.path.abspath('../Out/')

curr_image=0
timming=False
Save=False

diff=[]

def main():        
    for image_name in sorted(os.listdir(image_data_path)):
        start=time.time()
        image_path = os.path.join(image_data_path, image_name)
        image_in = cv2.imread(image_path)
        crop_lines,lines = crop_row_detect(image_in)
        if not timming:
            out_path=os.path.join((image_out_path)+image_name[:-4]+".PNG")
            
            #showing region of interest
            roi=[((85, 68),(225, 68)), ((225, 68),(319, 239)), ((0, 239),(319, 239)),((85, 68),(0, 239))] 
            for l in roi:
                cv2.line(image_in,l[0],l[1],(0,255,0)) 
            
            #showing crop rows detected by this approach
            with_crop_rows=cv2.addWeighted(image_in, 1, crop_lines, 3, 1.0)[68:,:]
            if Save:
                cv2.imwrite(out_path,with_crop_rows)#save the image 
            else: 
                cv2.imshow("crop-rows",with_crop_rows)
            while cv2.waitKey(1): # press any key to continue
                pass
			
        diff.append(time.time()-start)

    mean = 0
    for diff_time in diff: #finding the average time
        mean += diff_time

    #display timing 
    print('max time = {0}'.format(max(diff)))
    print('ave time = {0}'.format(float(mean) / len(diff)))

main()