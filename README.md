# Natural-Environment-Mushroom-Instance-Segmentation

This repository contains the source code from the research paper "Mask R-CNN in natural environments" and the data used for training. A subset of images have been extracted from [Google Open Images Dataset V6](https://storage.googleapis.com/openimages/web/factsfigures_v6.html). The subset contains images from the class "mushroom" and has been cleaned to contain only images in natural environments. 

 **This repository is not a complete tool for direct integration into other projects.**

## Installation
The installation uses Anaconda as the package manager in Ubuntu 22.04. Please follow these steps:

```bash
# Create the environment
conda create --name myenv python=3.8
conda activate myenv
```
 - If GPU with NVIDIA available:
	```
	conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia
	```
 - If only CPU available: 
	```
	conda install pytorch==1.13.0 torchvision==0.14.0 cpuonly -c pytorch
	```
	MMDetection library installation:
```
conda install -c conda-forge pycocotools
pip install -U openmim
mim install mmcv-full==1.7.0
wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.26.0.zip
unzip v2.26.0.zip
cd mmdetection-2.26.0/
pip install -v -e .
mim install mmengine==0.3.1
pip install yapf==0.40.1
```
Verify MMDetection installation (check result.jpg inside mmdetection folder):
```
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```


## Quickstart

Initially, the 5 dataset splits must be created using by executing the following python script:
```bash
cd code
python mushroom_datasets_merging.py
```
The pretrained weights of the models must be downloaded from the following list and moved in folder ***models_online_pretrained_weights***, without changing their name. It must be noted that the script is provided in its default publication configuration for a full training and inferencing run. However, there are many parameters can be changed inside the script for other customization.
| Model                                                        |
| ------------------------------------------------------------ |
| [Mask R-CNN ResNet50](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) |
| [Mask R-CNN ResNeXt101](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth) |
| [Mask R-CNN ConvNext](https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth) |
| [Mask R-CNN Swin (tiny)](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth) |
| [Mask R-CNN Swin (small)](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth) |
| [Mask Scoring R-CNN ResNet50](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth) |
| [Mask Scoring R-CNN ResNeXt101](https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth) |
| [Cascade Mask R-CNN ResNet50](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth) |
| [Cascade Mask R-CNN  ResNeXt101](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth) |
| [HTC ResNet50](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth) |
| [HTC ResNeXt101](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth) |
| [DetectoRS ResNet50](https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth) |

Inside natural_model_training.py the models to be used for training/inference can be selected by commenting or uncommenting lines (23-34). The choice of training using transfer learning from pretrained weights on COCO dataset or inference using already trained models is by setting variable ***use_online_pretrained*** = True/False respectively. Start training and inferencing:
```
cd code
python natural_model_training.py
```
The results will be saved inside "results" directory, which is automatically created by the script. Below is the folder structure after a complete training session. The performance metrics can be found inside eval_<id>.json or in terminal_output.txt
```
.
├── code/
├── data/
├── mmdetection_configs/
├── models_online_pretrained_weights/
└── results/
    ├── results_full_1/
    │   ├── mask_rcnn_r50_fpn_1x_coco/
    │   │   ├── configs/
    │   │   │   ├── <custom config file saved during training (.py)>
    │   │   │   └── <model weights with best validation score during training (.pth)>
    │   │   ├── test_outputs/
    │   │   │   ├── predicted_images/
    │   │   │   │   └── <visualized predictions on test set with green color masks>
    │   │   │   └── test_script_analysis/
    │   │   │       ├── terminal_output.txt
    │   │   │       ├── output.pkl
    │   │   │       ├── eval_<id>.json
    │   │   │       └── <visualized predictions on test set with red color masks>
    │   │   └── training_outputs/
    │   │       ├── tf_logs/
    │   │       └── training_logs.log.json
    │   ├── htc_r50_fpn_1x_coco
    │   └── ...
    ├── results_full_2/
    ├── results_full_3/
    ├── results_full_4/
    └── results_full_5/
```
