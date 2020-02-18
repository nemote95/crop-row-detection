# crop-row-detection
This project detect crop rows for different types of plants and in different fields by taking th following steps:
  1.  Converting to HSV color space
  2.  Applying a threshold for green parts
  3.  Selecting region of interest
  4.  Transforming into bird's eye view
  5.  Obtaining the skeleton 
  6.  Clustering white pixels by Mean-shift
  7.  Fitting lines for each cluster 
  8.  Transforming the result back into perspective view (this step is only for displaying and aligning the detected lines on the original photos)
  
However, this approach produces extera lines for some imagaes and can not detect curved lines. The following ideas can address these issues:
  - Filtering lines: 
     - detecting the distance between each line and remove lines that are located whithin that distance
     - knowing the exact distance between each row in ground truth data in advance
  - Perspective correction
    - developing an adaptive perspective transformation 
  - Curved rows: 
    - Polynomial line fitting 
    - finding successive short straight lines and representing curves as a set of short lines. 
  - Reducing noises:
    - Weeds segmentation/detections based on texture features of the green areas 
