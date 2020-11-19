#!/bin/bash

# Download cfg and weight files
gdown https://drive.google.com/uc?id=1tCZfUPkpY-TOUxIpcDo3XtM-SboHVxPr
tar xvfz data.tgz
cd data

# Download image file
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Make an example model directory
mkdir models && cd models
mkdir yolov2
