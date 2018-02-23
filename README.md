# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/Visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./writeup_images/Test_image_1.jpg "Traffic Sign 1"
[image5]: ./writeup_images/Test_image_2.jpg "Traffic Sign 2"
[image6]: ./writeup_images/Test_image_3.jpg "Traffic Sign 3"
[image7]: ./writeup_images/Test_image_4.jpg "Traffic Sign 4"
[image8]: ./writeup_images/Test_image_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AkshathaHolla91/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

The link to my HTML extract is [here](https://github.com/AkshathaHolla91/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing  the plot of number of input samples per class 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I did not implement grayscaling since the accuracy reduced due to loss of information after grayscaling. Hence I decided to perform normalization.

I tried normalizing the image set using various normalization models such as cv2 normalise function which normalised the data in the range of 0 to 1 and also using MinMax normalization to normalise the data in range of 0.1 to 0.9. Later I tried the recommended normalization function (Imagedata-128)/128 which normalises the data in the range of -1 to 1 and got slightly better results compared to the other models and hence chose to continue with that.

As seen in the visualization bar chart many classes in the data set have very few samples. Augumenting these classes with additional data would improve the performance on these classes, but due to time constraints I have chosen not to implement image augumentation here.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|				                                |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten	            | output 400                                    |
| Fully connected 400x120| bias 120 ,output 120        				    |
| RELU					|				                                |
| Dropout				|	Keep probability 0.65			            |
| Fully connected 120x84| bias 84 ,output 84        				    |
| RELU					|				                                |
| Dropout				|	Keep probability 0.65			            |
| Fully connected 84x43 | bias 43, output 43        					|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used  Adam optimizer algorithm (which is an extension to stochastic gradient descent) in which the loss operation was the reduced mean of the cross entopy.

#### Hyper parameters 
* learning rate: 0.002
* Epochs: 40
* batch size: 128
* Drop out probability: 35%

The Hyper parameters used here helped me to achieve the desired accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.944
* test set accuracy of 0.92

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    
    The architecture that I tried first was the Lenet architecture which by default gave me a training accuracy of around 90% without any additional layers.The Lenet architecture is a straightforward and small architetcure and because of this can be run on CPUs for smaller data sets and provides decent validation and test accuracy to start with.

* What were some problems with the initial architecture?

    The initial problem that I faced after using lenet with the current data set was that the Validation accuracy that I got by using the basic model was just 90%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    In order to improve the accuracy I tried normalisation of the data set as mentioned above which increased the validation accuracy to around 94 to 95 %. I also made modifications to the architecture by adding 1 drop out layer each with keep probability 0f 0.65 after the activation of the first 2  fully connected layers in order to reduce overfitting. Further I also  increased the learning rate to 0.002 and the number of epochs to 40  to be able to view the effects of adding the drop out layers for reducing overfitting and improving accuracy. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Except the 4 th image all other images that I chose were classified quite well with the current model since the classes that they belonged to had substantial number of samples in them. The 4th image would be classified to a different class in some test iterations since the training samples for that class were low. This can be further corrected by Image augumentation ie. adding additional samples to classes in which the sample count is less by rotation , translation , zoom etc of existing images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)   					    | 
| Speed limit (60km/h)  | Speed limit (60km/h)						    |
| Keep right			| Keep right									|
| Turn right ahead	    | Turn right ahead					 		    |
| Ahead only			| Ahead only      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed limit (30km/h) (probability of approximately 1), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Speed limit (30km/h)						    | 
| 7.58556716e-18     	| Speed limit (50km/h)							|
| 2.36261715e-26		| Speed limit (80km/h)						    |
| 5.05153526e-33	    | End of speed limit (80km/h)					|
| 9.62749893e-34	    | Speed limit (60km/h)     						|


For the second image, the model is relatively sure that this is a Speed limit (60km/h) sign (probability of approximately 0.99), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99899387e-01        | Speed limit (60km/h)						    | 
| 1.00614554e-04     	| Speed limit (50km/h)							|
| 2.18205076e-21		| Speed limit (80km/h)						    |
| 4.06584981e-26	    | Ahead only					                |
| 8.89098224e-31	    | Bicycles crossing     						|

For the third image, the model is relatively sure that this is a Keep right sign (probability of approximately 1), and the image does contain a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Keep right						            | 
| 0.00000000e+00     	| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)						    |
| 0.00000000e+00	    | Speed limit (50km/h)					        |
| 0.00000000e+00	    | Speed limit (60km/h)     						|

For the fourth image, the model is relatively sure that this is a Turn right ahead sign (probability of approximately 1), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Turn right ahead						        | 
| 3.29565229e-11     	| Roundabout mandatory							|
| 5.29518920e-13		| Keep left						                |
| 3.05841416e-16	    | Ahead only					                |
| 1.11683187e-16	    | Go straight or left     						|

For the fifth image, the model is relatively sure that this is a Ahead only sign (probability of approximately 1), and the image does contain a Ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Ahead only						            | 
| 0.00000000e+00     	| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)						    |
| 0.00000000e+00	    | Speed limit (50km/h)					        |
| 0.00000000e+00	    | Speed limit (60km/h)     						|
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I have not worked on this section now but hope to do so in the future.


