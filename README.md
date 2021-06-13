# JEDI 
**J**etson **E**mbedded platform-target **D**eep learning **I**nference acceleration framework with TensorRT

JEDI is a simple framework to apply various parallelization techniques on tkDNN-based deep learning applications running on NVIDIA Jetson boards such as NVIDIA Jetson AGX Xavier and NVIDIA Jetson Xavier NX. 

The main goal of this tool is applying various parallelization techniques to maximize the throughput of deep learning applications. 

## Applied Deep Learning Acceleration Techniques
- Preprocessing parallelization
- Postprocessing parallelization
- Intra-network pipelining with GPU and DLA
- Stream assignment per each pipelining stage
- Intermediate buffer assignment between pipelining stages
- Partial network duplication
- INT8 quantization on pipelined networks
- Batch

## FPS Results

This result is based on the old version of this software. (The target version is [commit](https://github.com/cap-lab/jedi/tree/73d855ef102b02e4352cba11f8db06005b49d015) )
Test environment: NVIDIA Jetson AGX Xavier (MAXN mode with jetson_clocks), Jetpack 4.3

| Network |  Baseline GPU (FP16) | GPU with parallelization techniques (FP16) |  GPU + DLA pipelining (FP16) | 
| :------:  | :-----:  | :-----:  | :-----:  |
| Yolov2 relu | 74  | 193  | 291  |
| Yolov3 relu | 50  | 87  | 133  |
| Yolov4 relu | 43  | 73  | 90  |
| Yolov4tiny relu | 103  | 459  | 504  |
| CSPNet relu | 40  | 62  | 72  |
| Densenet+Yolo relu | 44  | 86  | 120 |

## Index 
- [Supported Platforms](#supported-platforms)
- [Prerequisite](#prerequisite)
- [How to Compile JEDI](#how-to-compile-jedi)
- [JEDI Configuration Paramters](#jedi-configuration-parameters)
- [How to Run JEDI](#how-to-run-jedi)
- [How to Add a New Application in JEDI](#how-to-add-a-new-application-in-jedi)
- [Supported and Tested Networks](#supported-and-tested-networks)
- [Reference](#reference)

## Supported Platforms

- NVIDIA Jetson boards are supported. (Tested on NVIDIA Jetson AGX Xavier and NVIDIA Jetson Xavier NX)

## Prerequisite

- Forked [tkDNN](https://github.com/urmydata/tkDNN)
- All dependencies required by tkDNN
- Jetpack 4.3 or higher
- libconfig++
- OpenMP

## How to Compile JEDI

After installing the forked version of tkDNN, compile the JEDI with the following commands.
```
git clone https://github.com/urmydata/tkDNN.git
mkdir build && cd build
cmake ..
make
```

## How to Run JEDI

- To run JEDI, the following parameters are needed.
```
./build/bin/proc -c <JEDI configuration file> -r <JSON result file> -p <tegrastats log> -t <inference time output file>
```

where
  - `-c <JEDI configration file>`: JEDI configuration file (explanation of JEDI configuration file is shown in [here](#jedi-configuration-paramters))
  - `-r <JSON result file>` (optional): Output file of detection results in COCO JSON format.
  - `-p <tegrastats log output file>` (optional): Tegrastats log output file during inference which is used for computing the utilization and power.
  - `-t <inference time output file>` (optional): The output file which contains the total inference time

- Example commands of running JEDI
```
./build/bin/proc -h                                              # print help message
./build/bin/proc -c sample.cfg -r result.json -p power.log       # an example of running
```

## JEDI Configuration Parameters

- JEDI configuration file is based on [libconfg](https://hyperrealm.github.io/libconfig/) format.
- [sample.cfg](sample.cfg) is a sample configuration file with detailed explanation of each configuration parameters

## How to Add a New Application in JEDI

- JEDI provides an inteface to add a new tkDNN-based deep learning application.
- Currently, `YoloApplication` and `CenternetApplication` are implemented.
1. Write your own deep learning application with the [inference application implementation interface](src/inference_application.h)
  - `readCustomOptions`: Add a custom option which is used for this application.
  - `createNetwork`: Create a tkDNN-based network
  - `referNetworkRTInfo`: Refer NetworkRT class if any information in this class is needed
  - `initializePreprocessing`: Initialize preprocessing and input dataset
  - `initializePostprocessing`: Initialize postprocessing
  - `preprocessing`: Execute preprocessing 
  - `postprocessing`: Execute postprocessing (batched execution must be performed inside this method)
  - Call order: `readCustomOptions` => `createNetwork` => `referNetworkRTInfo` => `initializePreprocessing` => `initializePostprocessing` => `preprocessing`/`postprocessing`
  - You can also implement your own dataset with [dataset implementation interface](src/dataset.h)
2. Register your application with the following code in your source code.
```
REGISTER_JEDI_APPLICATION([Your application class name]);
```
3. Add your source code to [CMakeLists.txt](src/CMakeLists.txt)
4. Insert `app_type = "[Your application class name]"` in the JEDI configuration file.

## Supported and Tested Networks

| Network                         | Trained Dataset                             |
| :------------------------------ | :-----------------------------------------: |
| YOLO v2                         | COCO 2014 trainval                          |
| YOLO v2 tiny                    | COCO 2014 trainval                          |
| YOLO v3                         | COCO 2014 trainval                          |
| YOLO v3 tiny                    | COCO 2014 trainval                          |
| Centernet (DLA34 backend)       | COCO 2017 train                             |
| Cross Stage Partial Network     | COCO 2014 trainval                          |
| Yolov4                          | COCO 2014 trainval                          |
| Yolov4 tiny                     | COCO 2014 trainval                          |
| Scaled Yolov4                   | COCO 2017 train                             |
| Densenet+Yolo                   | COCO 2014 trainval                          |


