# XRDrive-Sim
Platform | Build Status |
-------- | ------------ |
Anaconda | [![Build status](https://ci.appveyor.com/api/projects/status/swutsp1bjcc56q64/branch/master?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/hand-tracking-samples/branch/master)

## Audience

The code in this repository is authored for computer-vision and machine-learning students and researchers interested in developing hand gesture applications using Intel® RealSense™ D400 depth cameras. The CNN implementation provided is intended to be used as a reference for training nets based on annotated ground-truth data; researchers may instead prefer to import captured datasets into other frameworks such as Google's TensorFlow and Caffe and optimize these algorithims using Model Optimizer from OpenVINO Toolkit.

This project provides Python code to demonstrate Hand Gestures via on PC camera or depth data, namely Intel® RealSense™ depth cameras. Additionally, this project showcases the utility of convolutional neural networks as a key component of real-time hand tracking pipelines using Intel OpenVINO. Two demo Jupyter Notebooks are provided showing hand gesture control from a webcam and depth camera. A YouTube video demonstrating some functionality the XRDrive code can be found: 

The software provided here works with the currently available Intel® RealSense™ D400 depth cameras supported by librealsense2. XRDrive is experimental code and not an official Intel® product.

## XRDrive Sim Notebooks and Python Samples

* [XRDrive_RealSense](xr_drive_realsense) - This application demonstrates basic dynamics-based hand tracking using synthetic data produced from a 3D model rendered to an OpenGL context (and corresponding depth buffer). This application is a good entrypoint into the codebase as it does not require any specific camera hardware. 

* [XRDrive_Cam](xr_drive_cam) - This sample introduces the dynamics-based tracker in the context of real-time input from an Intel® RealSense™ depth camera. It requires a RealSense™ SR300 device. Moreover, the sample illustrates the impact of real-world sensor noise and the need for effective segmentation. 

