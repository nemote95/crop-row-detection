# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:59:22 2018

@author: Negar
"""


import cv2,os,time
import numpy as np
import math 
from detect import crop_row_detect

### Setup ###
image_data_path = os.path.abspath('../CRBD/Images')
image_out_path = os.path.abspath('../Out/')

curr_image=0
timing=False

def main():        
    for image_name in sorted(os.listdir(image_data_path)):
        print (image_name)
        global curr_image
        curr_image += 1
 
        image_path = os.path.join(image_data_path, image_name)
            
        image_in = cv2.imread(image_path)
        crop_lines,lines = crop_row_detect(image_in)
            
        if timing == False:
            cv2.destroyAllWindows()
            cv2.imshow(image_name, cv2.addWeighted(image_in, 1, crop_lines, 3, 1.0))
            #cv2.imshow("blal",cv2.cvtColor(crop_lines,cv2.COLOR_GRAY2BGR))
    
            #out_path=os.path.join((image_out_path)+image_name[:-4]+".PNG")
            #print(out_path)
            #cv2.imwrite(out_path,image_in)
            
                
            print('Press any key to continue...')
            while cv2.waitKey(1) < 0:
                pass
        
            cv2.destroyAllWindows()

main()