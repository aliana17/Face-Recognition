# Face-Recognition

# Introduction

A python Program that identifies user using LBPH algorithm and perform task only when the user is authenticated. A simple Linear Regression model is created which is trained over the data set provided by the user using an IP Webcam© App.

# How it Works?

Firstly Samples of data are collected using Collecting_Data.py. The file collects over 100 samples images of the face using IP Webcam over the same network. These samples are saved in a directory path given in the file. 

For the above code to run you should have haarcascade_frontalface_default.xml file in the same directory where Collecting_Data.py and Model.py files are kept.

Model.py is used to train the model using the samples collected. After successfully training model, it will predict the authenticated face with high accuracy. 


