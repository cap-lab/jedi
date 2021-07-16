# JEDI 
**J**etson-aware **E**mbedded **D**eep learning **I**nference acceleration framework with TensorRT

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

- Test environment: NVIDIA Jetson AGX Xavier (MAXN mode with jetson_clocks), Jetpack 4.3
- Input image size: 416x416

### FP16

| Network |  Baseline GPU | GPU with JEDI |  GPU + DLA with JEDI | 
| :------:  | :-----:  | :-----:  | :-----:  |
| Yolov2 relu | 78  | 177 | **289** |
| Yolov2tiny relu | 97 | 566 | **618** |
| Yolov3 relu | 51 | 87 | **132** |
| Yolov3tiny relu | 110 | 580 | **670** |
| Yolov4 relu | 46 | 83 | **123** |
| Yolov4tiny relu | 111 | **596** | **602** |
| Yolov4csp relu | 42 | 94 | **142** |
| CSPNet relu | 41 | 64 | **80** |
| Densenet+Yolo relu | 46 | 86 | **119** |

### INT8

| Network |  Baseline GPU | GPU with JEDI |  GPU + DLA with JEDI | 
| :------:  | :-----:  | :-----:  | :-----:  |
| Yolov2 relu | 97 | 395 | **485** |
| Yolov2tiny relu | 103 | **663** | 612 |
| Yolov3 relu | 70 | 167 | **233** |
| Yolov3tiny relu | 119 | **762** | 672 |
| Yolov4 relu | 61 | 158 | **208** |
| Yolov4tiny relu | 116 | **728** | 672 |
| Yolov4csp relu | 50 | 177 | **236** |
| CSPNet relu | 66 | **149** | **150** |
| Densenet+Yolo relu | 62 | 183 | **225** |

## FPS Results (Old)

This result is based on the old version of this software. (The target version is [commit](https://github.com/cap-lab/jedi/tree/73d855ef102b02e4352cba11f8db06005b49d015) )

- Test environment: NVIDIA Jetson AGX Xavier (MAXN mode with jetson_clocks), Jetpack 4.3
- Input image size: 416x416

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
- [References](#references)

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

| Network                                | Trained Dataset      | Input size | Network cfg | Weights |
| :------------------------------------: | :------------------: | :--------: | :---------: | :-----: |
| YOLO v2<sup>1</sup> with relu                      | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo2_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo2_relu.zip) |
| YOLO v2 tiny<sup>1</sup> with relu                 | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo2tiny_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo2tiny_relu.zip) |
| YOLO v3<sup>2</sup> with relu                      | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo3_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo3_relu.zip) |
| YOLO v3 tiny<sup>2</sup> with relu                 | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo3tiny_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo3tiny_relu.zip) |
| Centernet<sup>4</sup> (DLA34 backend)              | COCO 2017 train      |  512x512  | - | [weights](https://cloud.hipert.unimore.it/s/KRZBbCQsKAtQwpZ/download) |
| Cross Stage Partial Network<sup>7</sup> with relu  | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/csresnext50-panet-spp_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/cspresnext_relu.zip) |
| Yolov4<sup>8</sup> with relu                       | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4_relu.zip) |
| Yolov4 tiny<sup>8</sup> with relu                  | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4tiny_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4tiny_relu.zip) |
| Scaled Yolov4<sup>10</sup> with relu                | COCO 2017 train      |  512x512  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolov4-csp_relu.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolov4csp_relu.zip) |
| Densenet+Yolo<sup>9</sup> with relu                | COCO 2014 trainval   |  416x416  | [cfg](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/densenet201_yolo.cfg) | [weights](https://github.com/cap-lab/jedi/releases/download/jedi_legacy/densenet_yolo_relu.zip) |

## References

1. Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
3. Yu, Fisher, et al. "Deep layer aggregation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
4. Zhou, Xingyi, Dequan Wang, and Philipp Krähenbühl. "Objects as points." arXiv preprint arXiv:1904.07850 (2019).
5. Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
6. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
7. Wang, Chien-Yao, et al. "CSPNet: A New Backbone that can Enhance Learning Capability of CNN." arXiv preprint arXiv:1911.11929 (2019).
8. Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934 (2020).
9. Bochkovskiy, Alexey, "Yolo v4, v3 and v2 for Windows and Linux" (https://github.com/AlexeyAB/darknet)
10. Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "Scaled-YOLOv4: Scaling Cross Stage Partial Network." arXiv preprint arXiv:2011.08036 (2020).




