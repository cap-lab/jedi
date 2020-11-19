# jedi
Jetson embedded platform-target deep learning inference acceleration framework with TensorRT

Jedi is a simple framework for fast inference on NVIDIA boards (NVIDIA Jetson AGX Xavier, Xavier NX).
It is possible to pipeling networks by spliting networks and mapping on different processing elements (GPU or NPUs).

## Prerequisite
We use the forked version of tkDNN [1] since the original tkDNN library does not support the pipelining.
So, please install the forked tkDNN on the [here](https://drive.google.com/file/d/1z_FXrien2twAH9Ic76Ep4pBfL4EsiUaq/view?usp=sharing, "tkdnn link")
The installation guide of the forked tkDNN is on the compressed file.

## Build
After install the modified version of tkDNN, now build this program as below.
```
$ mkdir build && cd build
$ cmake ../
$ make
$ cd ../
```
The binary file of the program is located in build/bin directory.

## Execution
Networks which use leaky or mish activation can not be run on DLA. 
So, we re-trained networks after changing such activations to relu activations.

Data can be download by typing as below.
```
$ pip install gdown                                                # we use gdown to download files
$ ./setting.sh
```
If you do not want to use gdown, then download in [here](https://drive.google.com/file/d/1tCZfUPkpY-TOUxIpcDo3XtM-SboHVxPr/view?usp=sharing, "data link")

To run the program, just type as below.
```
$ ./build/bin/proc -h                                              # print help message
$ ./build/bin/proc -c config.cfg -r result.json -p power.log       # an example of running
```
You can grasp about configuration file on sample configuration file.
Please check the path of the files indicated in the configuration file.

We are going to add further explanation after uploading the github.

[1] M. Verucchi, G. Brilli, D. Sapienza, M. Verasani, M. Arena, F. Gatti, A. Capotondi, R. Cavicchioli, M. Bertogna, M. Solieri
"A Systematic Assessment of Embedded Neural Networks for Object Detection", in IEEE International Conference on Emerging Technologies and Factory Automation (2020)
