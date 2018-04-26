"""
Created on Tue Apr 10 12:04:28 2018
@author: Negar
"""
import cv2
import numpy as np 
from scipy.optimize import curve_fit
import pylab as pl
import sklearn.cluster as clstr

def crop_row_detect(image,plot):
    hsv_mask=hsv_thresholding(image)
    warp_prespective=remove_prespective(hsv_mask)
    skel=skeletonize(warp_prespective)
    X,Y,labels=cluster(skel)
    lines=find_lines(X,Y,labels,plot)
    inversed= inverse_prespective(lines)
    return inversed,lines

def hsv_thresholding(image):
    '''Inputs a RGB image and outputs a binery image of the green areas in HSV'''

    min_values = np.array([37, 0, 0],np.uint8)
    max_values = np.array([150, 255, 255],np.uint8)

    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, min_values, max_values)
    
    return mask

def skeletonize(image):
    '''Inputs a binerized image and outputs the skeleton of the image'''
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    return skel


def remove_prespective(image):
    '''removes prespective dirtortion'''
    pts_src = np.array([[85, 68],[225, 68], [0, 239], [319, 239]], dtype = "float32")
    pts_dst = np.array([[0, 0],[140, 0],[0, 171],[140, 171]], dtype = "float32")
    h= cv2.getPerspectiveTransform(pts_src, pts_dst)
    im_out = cv2.warpPerspective(image, h, (140,171))
    return im_out

def inverse_prespective(lines):
    '''transforms to prespective view'''
    overlay = np.zeros(shape=(240,320,3))
    for l in lines:
        cv2.line(overlay,l[0],l[1],(0,0,255),1)
        
    pts_src = np.array([[85, 68],[225, 68], [0, 239], [319, 239]], dtype = "float32")
    pts_dst = np.array([[0, 0],[140, 0],[0, 171],[140, 171]], dtype = "float32")
    h= cv2.getPerspectiveTransform(pts_dst, pts_src)
    im_out = cv2.warpPerspective(overlay, h, (320,240))
    return im_out.astype(np.uint8)



def cluster(image):
    '''Inputs bird's eye view skeleton and outputs clusters of the skeleton'''

    X,Y=np.nonzero(image)

    try :
        bandwidth = clstr.estimate_bandwidth(Y.reshape(-1, 1), quantile=0.15)
        ms = clstr.MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=15)
        kmeansoutput=ms.fit(Y.reshape(-1, 1))
    except :
        ms = clstr.MeanShift()
        kmeansoutput=ms.fit(Y.reshape(-1, 1))

    labels=kmeansoutput.labels_
    return X,Y,labels

def find_lines(X,Y,labels,plot=False):
    '''Inputs a clusters and outputs lines fittet to the clusters'''

    lines =[]
    coefficients=[]
    if plot:
        plotHandles = []
        pl.figure('Meanshift')
        pl.scatter(Y,X, c=labels)
    for i in range(len(np.unique(labels))):
        cluster_indices= np.where(labels == i)[0]
        cluster_xs=X[cluster_indices]
        cluster_ys=Y[cluster_indices]
        
        try:
            if cluster_xs[-1]-cluster_xs[0]>60 and len(cluster_xs)>20 :
                coeff=np.polyfit(cluster_xs,cluster_ys,1)
                coefficients.append(coeff)
                f = np.poly1d(coeff)
                lines.append(((int(f(0)),0),(int(f(319)),319)))
                if plot:
                    p, = pl.plot(f(cluster_ys),cluster_xs, '-')
                    plotHandles.append(p)
                    
        except:
            pass
    if plot : 
        pl.legend(plotHandles)
        pl.show()        
        
    return lines
