# XRDrive-Sim
Platform | Build Status |
-------- | ------------ |
Anaconda | [![Build status](https://ci.appveyor.com/api/projects/status/swutsp1bjcc56q64/branch/master?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/hand-tracking-samples/branch/master)

## Audience

The code in this repository is authored for computer-vision and machine-learning students and researchers interested in developing hand gesture applications using Intel® RealSense™ D400 depth cameras. The CNN implementation provided is intended to be used as a reference for training nets based on annotated ground-truth data; researchers may instead prefer to import captured datasets into other frameworks such as Google's TensorFlow and Caffe and optimize these algorithims using Model Optimizer from OpenVINO Toolkit.

This project provides Python code to demonstrate Hand Gestures via on PC camera or depth data, namely Intel® RealSense™ depth cameras. Additionally, this project showcases the utility of convolutional neural networks as a key component of real-time hand tracking pipelines using Intel OpenVINO. Two demo Jupyter Notebooks are provided showing hand gesture control from a webcam and depth camera. A YouTube video demonstrating some functionality the XRDrive code can be found: 

The software provided here works with the currently available Intel® RealSense™ D400 depth cameras supported by librealsense2. XRDrive is experimental code and not an official Intel® product.

## XRDrive Sim Notebooks and Python Samples

* [XRDrive_RealSense](xr_drive_realsense.ipynb) - This application demonstrates RealSense-based hand gesture to control keyboard and mouse while playing a racing game using DeepHandNet model

* [XRDrive_Cam](xr_drive_cam.ipynb) - This application demonstrates RealSense-based hand gesture to control keyboard and mouse while playing a racing game using DeepHandNet model

* [RealSense_Openpose](openpose.py) - This application demonstrates openpose.py using OpenVINO Toolkit and RealSense-based implementation of [opencv-dnn](https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py) re-implementaion from [Carnegie Mellon University OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)




# Experimental Setup 

1. Research previous and current work on Hand Gestures

2. Defining hand gestures to control keys on a keyboard and mouse.

3. Data Collection

4. Training deep learning model which can recognize hand gestures at real time.

5. Develop the XRDrive Game Scene



# XRDrive Sim Hardware and Software

XRDrive is an inference-based application supercharged with power-efficient Intel® processors and Intel® Processor Graphics on a laptop.

ZenBook UX430
Intel Core i7 8th Gen
16 GB DDR4 RAM
512 SSD Storage
Intel RealSense D435
Ubuntu 18.04 LTS
ROS2OpenVINO
Anacodonda: Jupyter Notebook 
MxNet
Pyrealsense2
Pymouse


# 1. Research previous and current work on Hand Gestures

Xinghao Chen: awesome-hand-pose-estimation
3DiVi: Nuitrack 3D Gestures
Eyesight Embedded Computer Vision
Carnegie Mellon University OpenPose: webcam
Microsoft hololens: 3D sensor
Microsoft hand tracking: 3D sensor
Leap Motion: 3D sensor
Mano Motion: smartphone camera
PilotBit Mobile Hand Tracking: 3D sensor

# 2. Defining hand gestures to control keys on a keyboard and mouse.

Hand Gesture control keys for the keyboard are assigned to the left hand. Hand Gesture control keys for the mouse are assigned to the right hand. 
The normal has 10 key features:

1. closed hand
2. open hand
3. thumb only
4. additional index finger
5. additional middle finger
6. folding first finger
7. folding second finger
8. folding middle finger
9. folding index finger
10. folding thumb

We divided the left hand side into three regions (left, middle and right region). We use 3, 4, 5 as control, as a result we have 9 controls and 135 gestures that cover the keyboard. For a right hand as a mouse, all 10 features as gestures , which cover the inputs from mouse. The center of finger is a mouse cursor. The mouse cursor of computer will track the center of the right hand.

# 3. Data Collection

To train the DeepHandNet, big data is required and the data are to be collected by ourselves. We collected data from a variety of source including our own captured images, 2000 hand images were collected.




 
# 4. Training deep learning model which can recognize hand gestures at real time.
	## Training the MxNet DeepHandNet
Model is trained by 64x64 pixel images using Intel Architecture  
Data Preprocessing was done using OpenCV and Numpy to generate masked hand images. run xr_preprocessing_data.ipynb 
To recreate the Model run xrdrive_train_model.ipynb, It will read the preprocessed hand dataset, mask dataset and train the model, 400 epochs iterated.




