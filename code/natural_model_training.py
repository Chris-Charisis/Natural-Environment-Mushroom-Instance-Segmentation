import ast
import copy
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil

import torch
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, inference_detector, show_result_pyplot,init_detector


## Set up project folder
project_folder = "../"

## Uncomment which architectures will be examined
architectures_config_paths_dict = {
    # "mask_rcnn_r50_fpn_1x_coco" : project_folder + "mmdetection_configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
    # "mask_rcnn_x101_32x8d_fpn_1x_coco" : project_folder + "mmdetection_configs/mask_rcnn/mask_rcnn_x101_32x8d_fpn_1x_coco.py",
    # "ms_rcnn_r50_caffe_fpn_1x_coco" : project_folder + "mmdetection_configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py",
    # "ms_rcnn_x101_64x4d_fpn_1x_coco" : project_folder + "mmdetection_configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco.py",
    # "cascade_mask_rcnn_r50_fpn_1x_coco" : project_folder + "mmdetection_configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py",
    # "cascade_mask_rcnn_x101_64x4d_fpn_1x_coco" : project_folder + "mmdetection_configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py",
    # "htc_r50_fpn_1x_coco" : project_folder + "mmdetection_configs/htc/htc_r50_fpn_1x_coco.py",
    # "htc_x101_64x4d_fpn_16x1_20e_coco" : project_folder + "mmdetection_configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py",
    "detectors_htc_r50_1x_coco" : project_folder + "mmdetection_configs/detectors/detectors_htc_r50_1x_coco.py",
    # "mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco" : project_folder + "mmdetection_configs/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco.py",
    # "mask_rcnn_swin-t-p4-w7_fpn_1x_coco" : project_folder + "mmdetection_configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    # "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco" : project_folder + "mmdetection_configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",    
}

## Use all of the five splits of the train-validation sets
for dataset_num in ["full_1","full_2","full_3","full_4","full_5"]:
    ## Examine each architecture
    for architecture_selected in architectures_config_paths_dict.keys():
        variable_to_find_py_config_file = architecture_selected
        architecture_selected += "/"

        ## Folder structure creation
        working_folder = project_folder + "results/results_" + dataset_num + "/" + architecture_selected
        online_pretrained_weights_folder = project_folder + "models_online_pretrained_weights/"

        training_outputs = working_folder + "training_outputs/"
        test_outputs = working_folder + "test_outputs/"
        predicted_images = test_outputs + "predicted_images/"
        predicted_numerical_data = test_outputs + "predicted_numerical_data/"
        configs_folder = working_folder + "configs/"
        results_folder_path = "../results/results_" + str(dataset_num) + "/"

        ## Check whether the specified path exists or not
        os.makedirs(working_folder,exist_ok=True)
        os.makedirs(training_outputs,exist_ok=True)
        os.makedirs(configs_folder,exist_ok=True)
        os.makedirs(test_outputs,exist_ok=True)
        os.makedirs(predicted_images,exist_ok=True)
        os.makedirs(predicted_numerical_data,exist_ok=True)

        ## Set the baseline configuration and pre-trained weights
        architecture_config_file = architectures_config_paths_dict[variable_to_find_py_config_file]
        files = os.listdir(online_pretrained_weights_folder)
        architecture_pretrained_file = online_pretrained_weights_folder + [x for x in files if x.startswith(variable_to_find_py_config_file) and x.endswith(".pth")][0]

        ## Parameters for training and inference
        dataset_to_be_used = "natural"
        ## True to train the model or False to load pretrained model and do inference
        use_online_pretrained = True
        config_save_filename = "custom_config_" + architecture_selected[:-1]
        training_validation = True
        ## Visualization threshold
        prediction_images_threshold = 0.9
        ## Save prediction matrices in pickle format
        save_pickle_files = False
        if save_pickle_files:
            ## Threshold for confidence of prediction
            ## predictions with lower confidence values will not be saved in pickle file
            confidence_filtering_threshold = 0.9


        if use_online_pretrained:
            print("Using official pretrained model configuration and weights with transfer learning")
            print("--------------------------------------------------------------------------------")
            #select model config file and pretrained data
            cfg = Config.fromfile(architecture_config_file)
            cfg.load_from = architecture_pretrained_file

            ## Modify dataset type and path
            cfg.dataset_type = 'COCODataset'

            ####### Setting up dataset paths #######
            ##----------------------------------------------------------------------------------------------------------------------------------------
            if dataset_to_be_used=="natural":
                ## Google V6 natural env mushroom train set
                cfg.data.train.ann_file = project_folder +'data/' + dataset_num + '/train/annotations/' + dataset_num + '_train_instances.json'
                cfg.data.train.img_prefix = project_folder + 'data/' + dataset_num + '/train/images/'
                cfg.data.train.classes = ('mushroom',)
                cfg.data.train.seg_prefix = project_folder + 'data/' + dataset_num + '/train/images/'
                cfg.data.train.seg_suffix = ".jpg"        
                ## Google V6 natural env mushroom validation set
                cfg.data.val.ann_file = project_folder + 'data/' + dataset_num + '/validation/annotations/' + dataset_num + '_validation_instances.json'
                cfg.data.val.img_prefix = project_folder + 'data/' + dataset_num + '/validation/images/'
                cfg.data.val.classes = ('mushroom',)
                ## Google V6 natural env mushroom test set        
                cfg.data.test.ann_file = project_folder + 'data/test/annotations/test_instances.json'
                cfg.data.test.img_prefix = project_folder + 'data/test/images/'
                cfg.data.test.classes = ('mushroom',)


            else:
                pass
                ### Placeholder for a different future dataset 


            ####### Modifying default configuration #######
            ##----------------------------------------------------------------------------------------------------------------------------------------
            ## Modify num classes of the model in box head and mask head 
            cfg.data.samples_per_gpu = 1
            cfg.data.workers_per_gpu = 1

            ## Set up working dir to save files and logs.
            cfg.work_dir = training_outputs

            ## Set up learning rate
            cfg.optimizer.lr = 0.0001
            cfg["lr_config"]["step"] = [10]
            cfg.lr_config.warmup = None
            cfg.log_config.interval = 20

            ## Set up how often checkpoints are recorded
            evaluation_interval = 2
            ## Evaluation metrics used
            cfg.evaluation = dict(metric=['bbox', 'segm'], interval=evaluation_interval)
            ## Set the evaluation interval to reduce the evaluation times
            cfg.evaluation.interval = evaluation_interval
            ## Set the checkpoint saving interval to reduce the storage cost
            cfg.checkpoint_config.interval = evaluation_interval

            ## Set up total number of training epochs
            cfg["runner"]["max_epochs"] = 20

            ## Set seed thus the results are more reproducible
            cfg.seed = 0
            random.seed(0)
            np.random.seed(0)
            cfg.gpu_ids = range(1)

            ## Use tensorboard to log the training process
            cfg.log_config.hooks = [
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook')]

            ## Set up execution in GPU
            cfg["device"] = 'cuda'

            ## MAKE COMPATIBLE WITH THE 1 CLASS PROBLEM WE HAVE AT HAND
            ## FOR MASK R-CNN 
            if architecture_selected.startswith("mask"):
                print("Mask R-CNN case")
                cfg.model.roi_head.bbox_head.num_classes = 1
                cfg.model.roi_head.mask_head.num_classes = 1
                if architecture_selected.endswith("3x_coco/"):
                    cfg.fp16 = dict(loss_scale=512.)
                    cfg.runner.meta = dict(fp16=cfg.fp16)

            ## FOR MASK-SCORING R-CNN 
            elif architecture_selected.startswith("ms"): 
                print("Mask Scoring R-CNN case")
                cfg.model.roi_head.bbox_head.num_classes = 1
                cfg.model.roi_head.mask_head.num_classes = 1
                cfg["model"]["roi_head"]["mask_iou_head"]["num_classes"] = 1
 
            ## FOR CASCADE MASK R-CNN 
            elif architecture_selected.startswith("cascade"):
                print("Cascade Mask R-CNN case")
                cfg["model"]["roi_head"]["bbox_head"][0]["num_classes"] = 1
                cfg["model"]["roi_head"]["bbox_head"][1]["num_classes"] = 1
                cfg["model"]["roi_head"]["bbox_head"][2]["num_classes"] = 1
                cfg["model"]["roi_head"]["mask_head"]["num_classes"] = 1        


            ## FOR HTC OR DetectoRS
            elif architecture_selected.startswith("htc") or architecture_selected.startswith("detectors"):
                print("HTC or DetectoRS case")
                cfg["model"]["roi_head"]["bbox_head"][0]["num_classes"] = 1
                cfg["model"]["roi_head"]["bbox_head"][1]["num_classes"] = 1
                cfg["model"]["roi_head"]["bbox_head"][2]["num_classes"] = 1
                cfg["model"]["roi_head"]["mask_head"][0]["num_classes"] = 1  
                cfg["model"]["roi_head"]["mask_head"][1]["num_classes"] = 1  
                cfg["model"]["roi_head"]["mask_head"][2]["num_classes"] = 1  
                cfg["train_pipeline"][-1]["keys"] = ['img', 'gt_bboxes', 'gt_labels', 'gt_masks']
                cfg["data"]["train"]["pipeline"][-1]["keys"] = ['img', 'gt_bboxes', 'gt_labels', 'gt_masks']
                cfg["train_pipeline"][1]["with_seg"] = False
                cfg["data"]["train"]["pipeline"][1]["with_seg"] = False
                cfg["model"]["roi_head"].pop("semantic_roi_extractor")
                cfg["model"]["roi_head"].pop("semantic_head")        

            else:
                raise("Couldn't recognize the architecture provided")

            ## Look at the final config used for training
            # print(f'Config:\n{cfg.pretty_text}')

            ## SAVE CONFIG FILE TO .PY FILE
            text_file = open(configs_folder + config_save_filename + ".py", "w")
            text_file.write(cfg.dump())
            text_file.close()

        else:    
            ## LOAD CONFIG FILE FROM CUSTOM SAVED .PY FILE AND CHANGE THE PRETRAINED WEIGHT PATH
            architecture_config_file = configs_folder + [x for x in os.listdir(configs_folder) if variable_to_find_py_config_file in x and x.endswith(".py")][0]
            architecture_pretrained_file = configs_folder + [x for x in os.listdir(configs_folder) if variable_to_find_py_config_file in x and x.endswith(".pth")][0]
            cfg = Config.fromfile(architecture_config_file)
            cfg["load_from"] = architecture_pretrained_file

            print("Using custom pretrained model configuration file: ", architecture_config_file)
            print("Using custom pretrained model weights: ", architecture_config_file)
            print("---------------------------------------------------------")   


        ####### Build the configured model #######
        ##----------------------------------------------------------------------------------------------------------------------------------------
        if use_online_pretrained:    
            ## Build dataset
            datasets = [build_dataset(cfg.data.train)]

            ## Build the detector
            model = build_detector(cfg.model)

            ## Add an attribute for visualization convenience
            model.CLASSES = datasets[0].CLASSES

            train_detector(model, datasets, cfg, distributed=False, validate=training_validation,timestamp="training_logs")

            model.cfg = cfg

        else:
            ## build the model from a config file and a checkpoint file that exists inside the config as a path
            model = init_detector(cfg, cfg["load_from"], device='cuda')


        ####### Visualization on test set (green color) and predictions' numerical data collection #######
        ##----------------------------------------------------------------------------------------------------------------------------------------
        ## Set the test path 
        test_set_path = cfg.data.test.img_prefix
        test_set = sorted(os.listdir(test_set_path))
        ## Used dataset has image files only in jpg format
        test_set = [x for x in test_set if x.endswith("JPG") or x.endswith("jpg")]

        for test_img in test_set:
            ## Load image
            img = mmcv.imread(test_set_path + test_img)
            save_name = test_img.replace(".JPG", "_prediction.jpg")
            save_name = test_img.replace(".jpg", "_prediction.jpg")

            ## Model inference on img 
            result = inference_detector(model, img)

            if save_pickle_files:
                ## Filter out low confidence prediction in order to save up space in pickle files
                predictions_to_delete = []
                ## iterate through all existing pairs of predictions
                for idx in range(len(result[0][0])):
                    if result[0][0][idx][-1]<confidence_filtering_threshold:
                        predictions_to_delete.append(idx)

                predictions_to_delete.sort(reverse=True)
                if predictions_to_delete:
                    for index in predictions_to_delete:
                        ## delete the bounding box information
                        result[0][0] = np.delete(result[0][0],index,axis=0)
                        ## delete the mask array
                        del result[1][0][index]


                save_data_with_pickle(predicted_numerical_data + "/" + test_img.replace(".jpg", "_data.pkl") ,'result')

            ## Visualization of prediction for img and save the file with the overlayed masks on the initial image
            show_result_pyplot(model, img, result, out_file = predicted_images + save_name,score_thr=prediction_images_threshold)
            torch.cuda.empty_cache()
        

        ####### Find the best performing checkpoints using validation score #######
        ##----------------------------------------------------------------------------------------------------------------------------------------
        ## If no training is happening then no checkpoints will be produced from just inference
        if use_online_pretrained:
            ## Keep the checkpoint with the best validation score as final model and delete the other checkpoint to save space
            training_outputs = results_folder_path + architecture_selected + "training_outputs/"
            ## This json has the training information per epoch and validation step
            with open(training_outputs + "training_logs.log.json") as file:
                df = pd.DataFrame([ast.literal_eval(line.rstrip()) for line in file])

            ## Find the checkpoint with the best validation score
            val_map = df.loc[df["mode"]=="val","segm_mAP"].reset_index(drop=True)
            val_map.index = 2*val_map.index  + 2 
            ## Copy it to the configuration folder
            shutil.copy(training_outputs + "epoch_" + str(val_map.idxmax()) + ".pth", results_folder_path + architecture_selected + "configs/" + variable_to_find_py_config_file + "_BEST_mAP_epoch_" + str(val_map.idxmax()) + ".pth")

            ## Delete the rest of checkpoints to free memory
            filtered_files = [file for file in os.listdir(training_outputs) if file.endswith(".pth")]
            for file in filtered_files:
                path_to_file = os.path.join(training_outputs, file)
                print(path_to_file)
                os.remove(path_to_file)

        
        ####### Prepare the system command for performance evaluation on test set, visualization of predictions with red color #######
        ##----------------------------------------------------------------------------------------------------------------------------------------
        ## Path to custom config file
        config_filepath = results_folder_path + architecture_selected + "/configs/" + "custom_config_" + variable_to_find_py_config_file + ".py"
        ## Path to best checkpoint found earlier
        checkpoint_filepath = results_folder_path + architecture_selected + "/configs/" + [x for x in os.listdir(results_folder_path + architecture_selected + "/configs/") if x.endswith(".pth")][0]
        ## Path to store the results of the performance evaluation
        test_script_analysis_folder = results_folder_path  + architecture_selected + "/test_outputs/test_script_analysis/"
        ## Create the folder if doesn't exist already
        os.makedirs(test_script_analysis_folder, exist_ok=True)
        ## Save terminal output to a file
        terminal_output_txt_filepath = test_script_analysis_folder + "terminal_ouput.txt"
        ## Evaluation perfomance metrics in a pkl file, HOWEVER a json file is also produced containing the information for easier usage
        output_pkl_filepath = test_script_analysis_folder + "outputs.pkl"
        ## Execute the command through the system
        os.system("python ./test.py " + config_filepath + " " + checkpoint_filepath + " --eval bbox segm --show-score-thr 0.5 --work-dir " + test_script_analysis_folder + " --show-dir " + test_script_analysis_folder + " --out " + output_pkl_filepath + " > " + terminal_output_txt_filepath)
        

        print("------------------------------------------------------------------------------------------------------------------------------------")

