# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths of 24, 36, 48, 64 and 64 (model.py lines 52-56) 

The model includes RELU layers to introduce nonlinearity (code line 52-56 and 59-61), and the data is normalized in the model using a Keras lambda layer (code line 50). 

The images are cropped to have useful information and ignore the rest using Keras Cropping2D layer. This makes training te model faster.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 58). 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 65).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used training set data to keep the car on the lane.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to understand and implement behavioral cloning.

My first step was to use a convolution neural network model similar to architecture published by NVDIA autonomous team. I thought this model might be appropriate because it is more powerful than LeNet and solves end to end learning for self driving cars by detecting road features with human steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

Then I did data augmentation i.e flipped images and steering meaurements

To combat the overfitting, I modified the model by introducting a dropout layer between the set of convolutional, maxpooling2D and the set of fully connected layers.
 
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I moved architecture from LetNet to NVDIA.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 48-62) consisted of a convolution neural network with the following layers and layer sizes ...

Input = 32x32x3 RGB image Convolution Input = 65x320x3 Output = 31x158x24 Convolution Input = 31x158x24 Output = 14x77x36 Convolution Input = 14x77x36 Output = 5x37x48 Convolution Input = 5x37x48 Output = 3x35x64 Convolution Input = 3x35x64 Output = 1x33x64 Flatten Input = 1x33x64 Output = 2112 Dropout 0.5 Dense(RELU) Input=2112 Output=100 Dense(RELU) Input=100 Output=50 Dense(RELU) Input=50 Output=10 Dense(RELU) Input=10 Output=1

#### 3. Creation of the Training Set & Training Process

I used the udacity training data to input images to my network. I also used left and right camera images along with corrected steering angle value. (steering correction = 0.2) This can help the vehicle recover if off-centred.

To augment the data set, I also flipped images and angles thinking that the model would be less biased.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
