# crop-row-detection
This project detects crop rows in different fields by using a novel approach.
this approach has simple steps :
  1.  Convert to HSV color space
  2.  Threshold green parts
  3.  Select region of interest
  4.  Transform into bird's eye view
  5.  Obtain skeleton 
  6.  Cluster by Mean-shift
  7.  Fit lines
  8.  Transform into perspective view
  
this approach produces extera lines sometimes and can not detect curved lines. However, the following ideas can address these issues:
  - Filtering lines: 
     -  detecting the distance between each line and remove lines that are not located in that distance
     - knowing the exact distance between each row in ground truth data 
  - Perspective correction
    - developing an adaptive perspective transformation 
  - Curved rows: 
    - Polynomial line fitting 
    - finding successive short straight lines and representing curves as a set of short lines. 
  - Weeds segmentation based on texture features of the green areas 
