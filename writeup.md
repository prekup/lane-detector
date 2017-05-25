# **Finding Lane Lines on the Road** 

Road lane line detection is now a fundamental driver's aid in most new car models from major manufacturers and thus is a great first task for Term 1 of Udacity Self-driving Car Nanodegree. 

The initial task to detect road lanes lines in good conditions turned out to be quite simple, but things got more complicated when trying to make things work well with the challenge video. In the end the solution was quite robust and worked even without smoothing. 

Personal goal was to reduce reliance on smoothing as much as possible and to make the lane detection pipeline as robust as possible, but without resorting to implementing research papers. I tried to make the solution interesting to reviewers as well. It definitely was for me and I got to experiment a lot with many different approaches and algorithms. For me the interesting aspect was how a really good feature detection was achieved using simple methods and by iteratively refining the road/lane states. This made it possible to look at the detection process in a phenomenological way and to try to counter issues one stage at a time. For this, in debug mode, the internals of the algorithm can be observed and intermediate steps visualized.

#### Goals
Main goals of the project are:
* create a car lane line detection pipeline that detects road lane lines;
* create a written report.
Additional goal:
* detect lane lines of the challenge video.

#### Scope
As in project 4 we will be doing advanced lane tracking, in this project we will limit solution to feature detection and result smoothing. 

#### Highlights
1. Vanishing point detection by RANSAC fitting lines detected by probabilisticHough line detector, turned out to be remarkably stable and even robust to tree shadows. It highlights the importance of texture analysis in lane detection.
2. Only three parameters have to be tracked as the road can be approximately modelled as a triangle, where three points are the vanishing points and the intersection points of the lanes with the bottom of the image.
2. Deciding which lines belong to which lane, was done using the detected vanishing point. This turned out to be more robust then just deciding it based on line slope, as this is not robust against strong shadows like in the "challenge video".
3. Iterative weighted linear square regression was used to fit lane lines, to calculate the vanishing point according to the lane lines, to iteratively refine the lane positions and to detect outliers.
4. Binary mask was created from color channels. Color channels were tuned so that they would contain as much of the lane as possible and little of other image features. No gray scale image was necessary. Also in preprocessing stage no blurring, sharpening, contrast enhancement was needed for the test images.
5. Asymmetric line coherence and distance filtering in relation to the lane line (i.e. sides of the lane are not equivalent to the filter). This was to counteract effect of images with strong shadows and color texture differences, as they tended to bias the fitting towards bottom of the image or to just one of the lanes.

#### Acknowledgements
This work owes insights to a lot of various sources, but among them: Udacity course materials, OpenCV documentation, existing attempts and tips by fellow students David Lichtenberg, Kirill Danilyuk, Kyle Martin, Kyle Stewart-Frantz, Alex Staravoitau.


## Reflection

### Description of the pipeline

<a href="http://www.youtube.com/watch?feature=player_embedded&v=1YhSza1ysZg" target="_blank"><img src="http://img.youtube.com/vi/1YhSza1ysZg/0.jpg" alt="Canny detected challenge video" width="720" height="480" border="10" /></a>

In the current pipeline implementation, lane detection pipeline became a road detection pipeline, because 
in the projective space, a road can be tracked using three points: VP(vp_x, vp_y), LL(x2, y\_max), RL(x2', y\_max), where VP - vanishing point, LL - left lane point intersecting bottom of image, RL - as in LL, but for the right lane, x2 - x-coordinate where lane intersects the bottom image border (x1 belongs to the start of the line), y\_max height of the image and y-coordinate of the image bottom border (y-axis is top-down). 
The road can be also modelled to an approximation as a triangle \<VPLLRL. This is quite good model to approximate the road near the car. Although, when displaying the lane lines, we cut off a tiny amount of the lane near the VP, so as to display it in the manner wished for.

To overcome intermittent detection failures in case of too harsh optical conditions, smoothing is applied to the x2 coordinate of each lane (in left and right lane detectors) and to the vanishing point (in the road detector). 

The pipeline can be roughly grouped into four phases: preprocessing, line detection, road detection, lane detection and smoothing, image annotation.

Our pipeline consists of the following major stages:
#### Preprocessing:
1. Transformation to HSL color space. RGB image was transformed into HSL color space because there white and gray color channels can be selected separately. White can be chosen if lightness
 is large and gray can be selected when saturation is low.
 <a href="http://www.youtube.com/watch?feature=player_embedded&v=lWbYE1MP-4I" target="_blank"><img src="http://img.youtube.com/vi/lWbYE1MP-4I/0.jpg"  alt="Challenge video in HLS color space" width="720" height="480" border="10" /></a>
2. Image segmentation to white, gray, yellow and light-yellow color channels. Separation of yellow channel was complicated because the images have a lot of dry grass. To reduce the amount of image included into the color channel, it yello lines were detected using a yellow and a light-yellow channels.
3. Features selected by colour channels were combined into a binary mask.
4. Noise reduction of the lane and road binary masks using morphological open transform with cross kernel. It reduced the amount of edges detected and also enhanced road features near the vanishing points. It also closed small holes in the bitmask.
<a href="http://www.youtube.com/watch?feature=player_embedded&v=QulLlIX85ec" target="_blank"><img src="http://img.youtube.com/vi/QulLlIX85ec/0.jpg"  alt="Challenge video in HLS color space" width="720" height="480" border="10" /></a>
Gaussian blurring, unsharp mask, contrast enhanced local histogram equalization, gray scale image combined with the bitmask, etc were tried, but in the end they were not needed.

#### Line detection:
1. Edge detection of the lane and road binary masks using Canny edge detection producing a set of edges.
<a href="http://www.youtube.com/watch?feature=player_embedded&v=EqBp1q1oWRk" target="_blank"><img src="http://img.youtube.com/vi/EqBp1q1oWRk/0.jpg" alt="Canny detected challenge video" width="720" height="480" border="10" /></a>
2. Line detection based on edges using probabilistic Hough transform producing a set of lines.
<a href="http://www.youtube.com/watch?feature=player_embedded&v=b1GgHqmoPIw" target="_blank"><img src="http://img.youtube.com/vi/b1GgHqmoPIw/0.jpg" alt="Hough lane detection challenge video" width="720" height="480" border="10" /></a>
#### Road detection:

1. Angular filtering of lines that are too horizontal or too vertical.

##### Vanishing point detection:
1. Conversion of filtered lines to Hough space, where x is slope and y is intercept.
2. Fitting Hough space points using RANSAC regressor and with a high residual threshold (120) and line lengths as weights. High threshold was needed for exploring larger solution space and proved to be essential for the performance of the vanishing point detection. Loss function was L1.
3. Outliers determined by RANSAC are removed from the line set (I confirmed that same outliers are actually removed by later filtering steps as well, but it is preferable to refine the line set as early as possible. 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=0aWgaiGCssY" target="_blank"><img src="http://img.youtube.com/vi/0aWgaiGCssY/0.jpg" alt="Hough lane detection challenge video" width="720" height="480" border="10" /></a>
#### Lane detection first phase:
Uses the previously detected vanishing point (VP) to classify lines to lanes:
##### Initial filtering:
1. All lines above vanishing point are filtered out.
2. Use the previously detected vanishing point (VP) to classify lines to lanes (i.e. if line intercept is to the left of VP, classify to left lane, if not, to right lane). 
3. Right lane line should have negative slope (reverse for our case, as y axis is in reverse), left lane line the reverse. Classification by slope is not reliable without vanishing point, because in high-contrast shadow and color areas false positives are easy to come by. 
##### Initial fitting:
1. If set of lines is not empty, do weighted least squares fitting of the line set and return slope and intercept of the fitted line. Fit procedure uses L2 loss function. Also beginning and end points of the lines are fitted, because this increases the influence of line angle on the fitting result vs fitting line centroids.
2. Calculate X2 coordinate and add to buffer and calculate running average of x2.
3. Line to lane coherence filter. Filters lines by angle to lane. When looking in the context of the y axis, we can think of the lines as vectors that come towards us. Then we can look at how lines go forward to intersect with the lane and also how lines exit an intersection with the lane. This is necessary to filter out highly textured areas and to choose lines that will favorably affect the line to lane fitting process.
4. Line to lane distance filter. Filters out lines that are too far away from the lane. Depending on which side the line is, the removal distance can be different. This enables us to clean up the area between lane lines and thus reduce influence of outliers.
#### Lane detection second phase:
1. Calculate lane vanishing point based on detected lanes. 
2. Apply to lines of each lanes, a vanishing point based line filter. Distance from vanishing point to the extended version of the line is calculated and too far away lines are removed. Also lines below the vanishing point are removed, if they intersect a chosen interval on y axis. This is because at the vanishing point a lot of textures and lines converge and are not really that helpful in discovering the actual lane line positions.
3. Fit again the set of lines to the lane, using the same procesured as in 4.2.1.
4. Apply lane coherence and distance filters again.
5. Fit again the set of lines to the lane, as in step 3.
6. Update the buffer of X2 coordinate, replacing the last one, while backtracking and recalculating the running average of X2.
7. Recalculate vanishing point of lanes and add it to the finite vanishing points buffer, while calculating the running average of the vanishing points.

Hough detected lines along with lanes: <a href="http://www.youtube.com/watch?feature=player_embedded&v=OmyBP37uh8o" target="_blank"><img src="http://img.youtube.com/vi/OmyBP37uh8o/0.jpg" alt="Lange and line detection of challenge video" width="720" height="480" border="10" /></a>

#### Annotate image with detected lanes.

Video: green lines belong to left lane, blue lines to right lane. Lanes are colored red. Blue circle is the vanishing point detected in the hough space of the lines. The pink circle is the vanishing point calculated from the intersection of lanes.
<a href="http://www.youtube.com/watch?feature=player_embedded&v=MKLiFE1GPXw" target="_blank"><img src="http://img.youtube.com/vi/MKLiFE1GPXw/0.jpg" alt="Hough lane detection challenge video" width="720" height="480" border="10" /></a>

Further more detailed information can be found in the comments of the source code.

### Shortcomings

One of the main lessons was the importance of texture analysis and of the features state tracking. Also the pipeline depends on many parameters, which were optimized based on the test data. In real-life the amount of possible detection situations is much larger and thus a more flexible approach would be needed. Also the smoothing is very primitive and is not a part of the features state model.

### Possible improvements

* Probabilistic tracking of the features, for example using a Kalman filter or some other Bayesian filter would make the pipeline more stable and also reduce the amount of fixed parameters and make it possible to incorporate previous knowledge into the model in a more intelligent manner.
* Vanishing point detection and filtering of lines for the lanes could all be done within the Hough transform algorithm. This would make the lane detection faster and more efficient.
* Besides boolean filtering, voting should also be used for deciding which lines are to be declared outlier and which not.