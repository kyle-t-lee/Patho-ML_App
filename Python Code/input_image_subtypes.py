# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:40:39 2019

@author: MP_lab_GPU
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image


#repo_dir = 'C:\\Users\MP_lab_GPU\Desktop\Senior Design 2019\Senior Design\'

# CHANGE THIS DIRECTORY TO THE ML BREAST CANCER TOTAL FILES FOLDER
repo_dir = r'C:\Users\joekh\Documents\GitHub\ML-Breat_Cancer_Classfier\\'


#%%
imsize = 256
loader = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

#model = models.resnet152(pretrained=True)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 2)
model = models.vgg16(pretrained=True)
#num_ftrs = model.classifier[6].in_features # model.classifier[0].out_features #model.fc.in_features
#model.classifier[6] = nn.Linear(num_ftrs, 8)

#CHANGE PATH TO MODEL PATH, CHANGE MAP LOCATION BASED ON GPU
model_state_dict_path = r"C:\Users\joekh\OneDrive\Desktop\Senior Design Code\model_2020_2021\model_02252021\vgg_subclass_model_state_dict_02252021.pt"
model.load_state_dict(torch.load(model_state_dict_path,map_location=torch.device('cpu')))
model.eval()
#CHANGE PATH LOCATION TO FOLDER WITH CROPPED MALIGNANT IMAGES
#malignant_path = r'C:\Users\joekh\Documents\GitHub\ML-Breat_Cancer_Classfier\images\Photos for Testing\CroppedImages Malignant\\'
#CHANGE PATH LOCATION TO FOLDER WITH CROPPED NONMALIGNANT IMAGES
#non_malignant_path = r'C:\Users\joekh\Documents\GitHub\ML-Breat_Cancer_Classfier\images\Photos for Testing\CroppedImages Non_Malignant\\'

#%%
#### THE FIRST ARGUMENT IS BENIGN, SECOND IS MALIGNANT
tp = [] #num correctly diagnosed malignant
fp = [] #num incorrectly diagnosed malignant
tn = [] #num correctly diagnosed benign
fn = [] #num incorrectly diagnosed negative
adenosis= []
ductal_carcinoma= []
fibroadenoma= []
lobular_carcinoma= []
mucinous_carcinoma= []
papillary_carcinoma= []
phyllodes_tumor= []
tubular_adenoma= []
errors = 0
num_malignant = 0
num_non_malignant = 0
subtype_correctness = 0

testing_folder_path = r"C:\Users\joekh\OneDrive\Desktop\Senior Design Code\Image_Dir\test"

for folder in os.listdir(testing_folder_path):
    for i in os.listdir(os.path.join(testing_folder_path,folder)): 
        try:
            image = image_loader(loader, os.path.join(testing_folder_path,folder, i))
            
            y = model(image)
            print(y.argmax().item())
            if y.argmax().item() == 0 or y.argmax().item() == 2 or y.argmax().item() == 6 or y.argmax().item() == 7: #if model classifies as benign subtypes
                if(folder == "adenosis" or folder == "fibroadenoma" or folder == "phyllodes_tumor" or folder == "tubular_adenoma"): # if it is in benign folders
                    tn.append(i)
                    num_non_malignant+=1
                else:
                    fn.append(i)
                    num_malignant+=1
                if y.argmax().item() == 0:
                    adenosis.append(i)
                    if folder == "adenosis":
                        subtype_correctness +=1
                elif y.argmax().item() == 2:
                    fibroadenoma.append(i)
                    if folder == "fibroadenoma":
                        subtype_correctness +=1
                elif y.argmax().item() == 6:
                    phyllodes_tumor.append(i)
                    if folder == "phyllodes_tumor":
                        subtype_correctness +=1
                else:
                    tubular_adenoma.append(i)
                    if folder == "tubular adenoma":
                        subtype_correctness +=1
            elif y.argmax().item() == 1 or y.argmax().item() == 3 or y.argmax().item() == 4 or y.argmax().item() == 5: # if model classifies as malignant subtype
                if(folder == "adenosis" or folder == "fibroadenoma" or folder == "phyllodes_tumor" or folder == "tubular_adenoma"): # if it is in benign folders
                    fp.append(i)
                    num_non_malignant+=1
                else:
                    tp.append(i)
                    num_malignant+=1
                if y.argmax().item() == 1:
                    ductal_carcinoma.append(i)
                    if folder == "ductal_carcinoma":
                        subtype_correctness +=1
                elif y.argmax().item() == 3:
                    lobular_carcinoma.append(i)
                    if folder == "lobular_carcinoma":
                        subtype_correctness +=1
                elif y.argmax().item() == 4:
                    mucinous_carcinoma.append(i)
                    if folder == "mucinous_carcinoma":
                        subtype_correctness +=1
                else:
                    papillary_carcinoma.append(i)
                    if folder == "papillary_carcinoma":
                        subtype_correctness +=1
            else:
                print(i)
                errors+=1
        except:
            print(i)
            errors +=1
    # for i in os.listdir(malignant_path):
    #     try:
    #         image = image_loader(loader, os.path.join(malignant_path, i))
            
    #         y = model(image)
    #         if y.argmax().item() == 0:
    #             fn.append(i)
    #         else:
    #             tp.append(i)
    #         num_malignant +=1
    #     except:
    #         print(i)
    #         errors +=1
            
    # for i in os.listdir(non_malignant_path):
    #     try:
    #         image = image_loader(loader, os.path.join(non_malignant_path, i))
    #         num_non_malignant +=1
    #         y = model(image)
    #         if y.argmax().item() == 0:
    #             tn.append(i)
    #         else:
    #             fp.append(i)
    #     except:
    #         errors +=1

print("True Positives: "+str(tp))
print("True Negatives: "+str(tn))
print("False Positives: "+str(fp))
print("False Negatives: "+str(fn))
print()
print("Benign: "+str(num_non_malignant))
print("Adenosis: "+str(adenosis))
print("Fibroadenoma: "+str(fibroadenoma))
print("Phyllodes Tumor: "+str(phyllodes_tumor))
print("Tubular Adenoma: "+str(tubular_adenoma))
print()
print("Malignant: "+str(num_malignant))
print("Ductal Carcinoma: "+str(ductal_carcinoma))
print("Lobular Carcinoma: "+str(lobular_carcinoma))
print("Mucinous Carcinoma: "+str(mucinous_carcinoma))
print("Papillary Carcinoma: "+str(papillary_carcinoma))
print()
print("Total Number of images: "+str(len(tp)+len(tn)+len(fp)+len(fn)))
print("Subtype Correctness: "+str(subtype_correctness))
print("Subtype Correctness %: "+str(100*subtype_correctness/(len(tp)+len(tn)+len(fp)+len(fn))))
print("True Positives: "+str(len(tp)))
print("True Negatives: "+str(len(tn)))
print("False Positives: "+str(len(fp)))
print("False Negatives: "+str(len(fn)))


            
    
    
#%%







