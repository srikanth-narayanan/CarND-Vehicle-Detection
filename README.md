[//]: # (Image References)
[image1]: ./output_images/draw_rectangles.png
[image3]: ./output_images/sampleset.jpg
[image4]: ./output_images/HOG_Vehicle.jpg
[image5]: ./output_images/HOG_nonvehicle.png
[image6]: ./output_images/feataure_scaling.png
[image7]: ./output_images/confusionmatrix.png
[image8]: ./output_images/slidingwindow.png
[image9]: ./output_images/heatmap.png
[video1]: ./project_video.mp4

# Vehicle Detection

This project consists of constructing a pipeline needed to identify vehicles in a video stream.

The goals / steps of this project are the following:

- Explore and understand the given dataset of car and not a car image dataset.
- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
- Train a classifier using the dataset to determine if a given image is car or not a car.
- Define a pipeline to process a given image to identify if it has a car.
- Estimate a bounding box for the vehicles detected.
- Reduce the false positives in the identification.


### Explore the dataset

As a first step i tried to explore and understand what does the given dataset contains, size of the images, visualise a few samples of car and not a car images. The following is a plot to show the samples of the images.

![Sample Dataset][image3]

A function to draw rectangle is defined in the function `draw_rectangle` in cell 11. The following is an example of box around a given region.

![Draw Rectangle][image1]


### Histogram of Oriented Gradients (HOG)

The HOG values of given image can act as a unique signature in identifying a specific appearance of a feature like car. The function to determine the hog feature of an image is defined by the fucntion `get_hog_features` in the notebook cell 7 and 8. The following is an example of HOG image of a car and not a car.

![HOG of a car][image4]

![HOG of a Not a car][image5]

A different combination of HOG parameter were evaluated as shown below in the table.

| Configuration | color space | orientation | pix per cell | cell per block | channel | Extract_Time |
|---------------|-------------|-------------|--------------|----------------|---------|--------------|
| 0             | YUV         | 9           | 8            | 2              | ALL     | 84.154645    |
| 1             | YUV         | 9           | 8            | 2              | 0       | 26.747634    |
| 2             | YUV         | 9           | 8            | 2              | 1       | 27.399855    |
| 3             | YUV         | 9           | 8            | 2              | 2       | 27.281057    |
| 4             | YUV         | 9           | 16           | 2              | ALL     | 49.047909    |
| 5             | YUV         | 9           | 16           | 2              | 0       | 16.573284    |
| 6             | YUV         | 9           | 16           | 2              | 1       | 16.791569    |
| 7             | YUV         | 9           | 16           | 2              | 2       | 17.329109    |

After finalising on the parameter, HOG extraction using open CV hog function was also evaluated. The extraction time of the openCV function was significantly faster for the same requirements.

| Configurations | color space | orientation | pix per cell | cell per block | channel | Extract_Time |
|----------------|-------------|-------------|--------------|----------------|---------|--------------|
| 0              | YUV         | 9           | 8            | 2              | ALL     | 5.052052     |
| 1              | YUV         | 9           | 16           | 2              | ALL     | 6.059657     |

In the final pipeline openCV HOG method was used to extract the features.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

