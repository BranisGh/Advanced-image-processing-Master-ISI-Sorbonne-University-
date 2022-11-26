# advanced-image-processing-Master-ISI-Sorbonne-University-

In this lab, we propose to implement and study the functioning (influence of parameters, limitations, variants) of a detector and a descriptor in order to realize a panorama application. The implementation is done in Python.    

### 1. DETECTION OF POINTS OF INTEREST  
The objective of this part is to implement and study two algorithms for detecting points of interest.   
  
1.1 Harris detector :  
(a) Implementation of the Harris detector using a rectangular and then Gaussian weighting window.    
(b) Implementation of a suppression of local non-maxima in order to avoid "clusters" of interest points.    
(c) Study of the influence of the type of weighting window, of its size and of the `"k" parameter in the detection of points of interest.    
(d) Use of existing Python functions to rotate the image. Adapt the parameters of the to have a rotation invariant detector.    

1.2 FAST detector :  
(a) Implementation of the FAST detector. To check if n consecutive points are 1, we use the convolution function of scipy of Python), that is convolve the vector by a filter of size n filled with 1.     
(b) Implementation of the removal of local non-maxima.     
(c) Using existing Python functions to rotate the image. Adapt the parameters of to have a rotation invariant detector.    

1.3 Comparison:  
(a) Comparison of interest points detected by the Harris detector and the FAST detector.    
(b) Use of existing Python functions to compare the detectors to those already implemented (Harris, FAST or others).    

### 2. DESCRIPTION AND MATCHING OF INTEREST POINTS  
The objective of this part is to implement and study a method of description and Matching of the points of interest detected in the previous exercise.  
  
2.1 Simple description: Using an intensity block    
(a) For each point of interest, we retrieve a block of pixels around it and transform it into a vector.   
(b) Implement a Matching method able to handle bad matches (either via cross-matching or distance comparison).        
(c) Study of the influence of the size of the block and the comparison metrics used on the quality of the matching. For the metrics, we will use the correlation (and we will study the interest of its normalization) and another metric of our choice.      
(d) Use existing Python functions to rotate one of the images. Implementing a solution to rig the matching problem.  

2.2 Description "advance
(a) Implement the LBP descriptor and combine it with your FAST descriptor.  
(b) Study the influence of the different parameters of this descriptor on the quality of the matches.  
(c) Use existing Python functions to compare the description to other more "advanced" ones (e.g. SIFT). 


### 3. PANORAMA (in progress ...)  
The objective of this exercise is to use the matches found in the previous exercise to merge the 2 images and form a panorama.    

3.1 Implementation of the DLT (direct linear transform) algorithm.   
3.2 Study of the influence of the normalization of the data on the calculation of the DLT.     
3.3 Use of the transformation provided by the DLT to determine the size of the panorama (transformation of the 4 corners of image 2 to the frame of image 1 for example) and then to project image 2 (and interpolate it) into the frame of image 1 to form the panorama.  




