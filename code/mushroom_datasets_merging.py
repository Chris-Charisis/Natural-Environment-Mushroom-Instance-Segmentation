import copy
import json
from natsort import natsorted
import os
import random
import shutil

with open("../data/combined_train_instances.json") as json_file:
    train_annotations = json.load(json_file)

target_directory = "../data/new_train/"

def delete_and_reindex_images(annotations,images_to_delete):
    #deleting selected images
    deleted_images_ids = []
    images_used = []
    images_UNused = []
    for idx,image in enumerate(annotations["images"]):
        if image["file_name"] in images_to_delete:
            deleted_images_ids.append(image["id"])
            images_UNused.append(image)
        else:
            images_used.append(image)

    annotations["images"] = images_used
    
    #reindexing remaining images and keeping the old/new index pairs for annotations reindexing
    images_reindexing_dict = {}
    
    for idx,image in enumerate(annotations["images"],1):
            images_reindexing_dict[image["id"]] = idx
            image["id"] = idx
        
    return annotations,images_reindexing_dict,deleted_images_ids

def delete_and_reindex_annotations(annotations,images_reindexing_dict,deleted_images_ids):
    annotations_used = []
    annotations_UNused = []
    for idx,annotation in reversed(list(enumerate(annotations["annotations"]))):
        if annotation["image_id"] in deleted_images_ids:
            annotations_UNused.append(annotation)
        else:
            annotations_used.append(annotation)

    annotations["annotations"] = annotations_used
    
    for idx,annotation in enumerate(annotations["annotations"],1):
        annotation["id"] = idx
        annotation["image_id"] = images_reindexing_dict[annotation["image_id"]]
        
    return annotations

## six cross-validation of full set
for i in range(1,6):
    random.seed(i)
    print(i)
    full_images_list = copy.deepcopy(train_annotations["images"])
    ## suffle the images based on random seed
    random.shuffle(full_images_list)
    #get the first 1100 for train and the last 130 for validation
    full_train = full_images_list[:1100]
    full_val = full_images_list[1100:]
    ## create lists of the filenames for train and validation
    ## they will used to delete the extra entries on the opposite set
    ## delete all validation images from initial set to create the train set and vice versa
    ## using the filenames as they remain unchanged in the whole procedure
    full_train_selected_images = []
    full_validation_selected_images = []

    for image in full_train:
        full_train_selected_images.append(image["file_name"])

    for image in full_val:
        full_validation_selected_images.append(image["file_name"])
        
    ## create train annotation
    images_to_delete = full_validation_selected_images
    full_train_annotations = copy.deepcopy(train_annotations)
    full_train_annotations, images_reindexing_dict, deleted_images_ids = delete_and_reindex_images(full_train_annotations,images_to_delete)
    full_train_annotations = delete_and_reindex_annotations(full_train_annotations,images_reindexing_dict,deleted_images_ids)   
    ## create validation annotation
    images_to_delete = natsorted(full_train_selected_images)
    full_val_annotations = copy.deepcopy(train_annotations)
    full_val_annotations, images_reindexing_dict, deleted_images_ids = delete_and_reindex_images(full_val_annotations,images_to_delete)
    full_val_annotations = delete_and_reindex_annotations(full_val_annotations,images_reindexing_dict,deleted_images_ids)
    ## creating the folders
    target_folder = "../data/full_" + str(i) + "/"
    os.makedirs(target_folder, exist_ok = True)
    os.makedirs(target_folder + "train/annotations/", exist_ok = True)
    os.makedirs(target_folder + "train/images/", exist_ok = True)
    os.makedirs(target_folder + "validation/annotations/", exist_ok = True)
    os.makedirs(target_folder + "validation/images/", exist_ok = True)
    ## copying the images
    for image in full_train_selected_images:
        shutil.copy(target_directory + image, target_folder + "train/images/" + image)
    for image in full_validation_selected_images:
        shutil.copy(target_directory + image, target_folder + "validation/images/" + image)
    ## saving the annotations
    with open(target_folder + "train/annotations/full_" + str(i) + "_train_instances.json", 'w') as fp:
        json.dump(full_train_annotations, fp)
    with open(target_folder + "validation/annotations/full_" + str(i) + "_validation_instances.json", 'w') as fp:
        json.dump(full_val_annotations, fp)        
    
    print("----------------")

