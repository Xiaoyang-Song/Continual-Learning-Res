import csv
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import operator

# os.chdir("U:\Research Projects\Incremental Learning with Raed\PointCloud Example")

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, TensorDataset, Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

# import pptk

from loguru import logger

from tqdm import tqdm
from EWC_utils import EWC, ewc_train, test, variable
import random

import incremental_dataset as incremental
from scoreslib.utils.exp import (
    get_model,
    get_transform,
    get_mean, 
    get_std,
    get_dataloader,
)
from scoreslib.inference.ODIN import search_ODIN_hyperparams, get_ODIN_score
from scoreslib.utils import split_dataloader
from scoreslib.metric import get_metrics, train_lr
from scoreslib.inference import get_feature_dim_list
from scoreslib.inference.Mahalanobis import (
        sample_estimator, 
        search_Mahalanobis_hyperparams,
        get_Mahalanobis_score_ensemble,
    )


#%% import data
indat_defect_raw_A = np.loadtxt(open("Defect_Surface_A_Labeled.csv", "rb"), delimiter=",")
indat_defect_raw_B = np.loadtxt(open("Defect_Surface_B_Labeled.csv", "rb"), delimiter=",")
indat_defect_raw_C = np.loadtxt(open("Defect_Surface_C_Labeled.csv", "rb"), delimiter=",")
indat_defect_raw_D = np.loadtxt(open("Defect_Surface_D_Labeled.csv", "rb"), delimiter=",")
indat_defect_raw = np.concatenate((indat_defect_raw_A, indat_defect_raw_B, indat_defect_raw_C, indat_defect_raw_D), axis = 0)
print(indat_defect_raw.shape)

indat_all_raw = np.loadtxt(open("ptC_mm.txt", "rb"), delimiter=",")
indat_normal_raw = np.loadtxt(open("Normal_Surface_Labeled.txt", "rb"), delimiter=",")

nrow_defect = len(indat_defect_raw)
nrow_normal = len(indat_normal_raw)

# merge small dent with hole
for i in range(indat_defect_raw_A.shape[0]):
    if indat_defect_raw_A[i, 3] == 3:
        indat_defect_raw_A[i, 3] = 1
    if indat_defect_raw_A[i, 3] > 3:
        indat_defect_raw_A[i, 3] = indat_defect_raw_A[i, 3] - 1
for i in range(indat_defect_raw_B.shape[0]):
    if indat_defect_raw_B[i, 3] == 3:
        indat_defect_raw_B[i, 3] = 1
    if indat_defect_raw_B[i, 3] > 3:
        indat_defect_raw_B[i, 3] = indat_defect_raw_B[i, 3] - 1
for i in range(indat_defect_raw_C.shape[0]):
    if indat_defect_raw_C[i, 3] == 3:
        indat_defect_raw_C[i, 3] = 1
    if indat_defect_raw_C[i, 3] > 3:
        indat_defect_raw_C[i, 3] = indat_defect_raw_C[i, 3] - 1
for i in range(indat_defect_raw_D.shape[0]):
    if indat_defect_raw_D[i, 3] == 3:
        indat_defect_raw_D[i, 3] = 1
    if indat_defect_raw_D[i, 3] > 3:
        indat_defect_raw_D[i, 3] = indat_defect_raw_D[i, 3] - 1


# generate 3D point cloud image data for 4 types of defects
n_defect = 5
n_img_defect = 10000 # number of images
defect_full_index = np.array(range(nrow_defect))
size_img = 300 # number of points
radius = 10 # ball radius for 3D point cloud sampling
search_range = 5000
img_predictor_defect = np.zeros((n_defect, n_img_defect, size_img, 3)) # store the image predictors
img_response_defect = np.zeros((n_defect, n_img_defect))
img_center_defect = np.zeros((n_defect, n_img_defect, 3))
for defect_type in tqdm(range(n_defect)):
    # find defect centers
    boolRow = indat_defect_raw[:, 3] == (defect_type + 1)
    i = 0
    while (i < n_img_defect):
        center_index = np.random.choice(defect_full_index[boolRow], size = 1, replace = True).squeeze()
        # find 3D point cloud cluster within ball radius from each ball center
        indat_defect_raw_partial = indat_defect_raw[max(0, center_index - search_range) : min(nrow_defect, center_index + search_range), 0 : 3]
        local_index = np.arange(0, len(indat_defect_raw_partial))[np.sum((indat_defect_raw_partial - np.tile(indat_defect_raw[center_index, 0 : 3], (len(indat_defect_raw_partial), 1))) ** 2, axis = 1) < radius ** 2]
        
        if len(local_index) >= size_img:
            # sample size_img points
            select_index = np.random.choice(local_index, size_img, replace = False).squeeze()
            # store point clouds, centers, and labels
            img_predictor_defect[defect_type, i, :, :] = indat_defect_raw_partial[select_index, 0 : 3] - np.tile(indat_defect_raw[center_index, 0 : 3], (size_img, 1))
            img_center_defect[defect_type, i, :] = indat_defect_raw[center_index, 0 : 3]
            img_response_defect[defect_type, i] = defect_type + 1            
            i += 1
        else:
            continue

    # random shuffle rows
    shuffled_index = np.arange(n_img_defect)
    np.random.shuffle(shuffled_index)
    img_predictor_defect[defect_type, :, :, :] = img_predictor_defect[defect_type, shuffled_index, :, :]
    img_response_defect[defect_type, :] = img_response_defect[defect_type, shuffled_index]
    img_center_defect[defect_type, :, :] = img_center_defect[defect_type, shuffled_index, :]

    # sort the last column
    for i in range(len(img_predictor_defect)):
        img_predictor_defect[defect_type, i, :, :] = np.array(sorted(img_predictor_defect[defect_type, i, :, :], key = operator.itemgetter(2, 1)))

# generate 3D point cloud image data for normal surface
n_img_normal = n_img_defect # number of images
normal_full_index = np.array(range(nrow_normal))
img_predictor_normal = np.zeros((n_img_normal, size_img, 3)) # store the image predictors
img_response_normal = np.zeros(n_img_normal)
img_center_normal = np.zeros((n_img_normal, 3))
# find defect centers
for i in range(n_img_normal):
    center_index = np.random.choice(normal_full_index, size = 1, replace = True).squeeze()
    # find 3D point cloud within each ball
    indat_normal_raw_partial = indat_normal_raw[max(0, center_index - search_range) : min(nrow_normal, center_index + search_range), 0 : 3]
    local_index = np.arange(0, len(indat_normal_raw_partial))[np.sum((indat_normal_raw_partial - np.tile(indat_normal_raw[center_index, 0 : 3], (len(indat_normal_raw_partial), 1))) ** 2, axis = 1) < radius ** 2]

    if len(local_index) >= size_img:
        # sample size_img points
        select_index = np.random.choice(local_index, size_img, replace = False).squeeze()
        # store point clouds, centers, and labels
        img_predictor_normal[i, :, :] = indat_normal_raw_partial[select_index, 0 : 3] - np.tile(indat_normal_raw[center_index, 0 : 3], (size_img, 1))
        img_center_normal[i, :] = indat_normal_raw[center_index, 0 : 3]
        i += 1
    else:
        continue

print(img_predictor_normal.shape)
print(img_response_normal.shape)
print(img_predictor_defect.shape)
print(img_response_defect.shape)
print(Counter(img_response_normal))
for cls_data in img_response_defect:
    print(Counter(cls_data))
# sort the last column
for i in range(len(img_predictor_normal)):
    img_predictor_normal[i, :, :] = np.array(sorted(img_predictor_normal[i, :, :], key = operator.itemgetter(2, 1)))

#% organize dataset as multiple stages
ntrain = 8000
ntest = 2000

# task lists
# 1 small dent, 2 corner crack, 3 big dent, 4 wood grain, 5 long crack
task_init = np.array([0, 1, 3])

train_data = {}
test_data = {}
multitask_train_loader = {}
multitask_test_loader = {}

# baseline dataset (InD)
temp_img_train = img_predictor_normal[0:ntrain, :, :]
temp_resp_train = img_response_normal[0:ntrain]
temp_img_test = img_predictor_normal[ntrain:(ntrain+ntest), :, :]
temp_resp_test = img_response_normal[ntrain:(ntrain+ntest)]
for i in task_init:
    temp_img_train = np.vstack((temp_img_train, img_predictor_defect[i, 0:ntrain, :, :]))
    temp_resp_train = np.append(temp_resp_train, img_response_defect[i, 0:ntrain])
    temp_img_test = np.vstack((temp_img_test, img_predictor_defect[i, ntrain:(ntrain+ntest), :, :]))
    temp_resp_test = np.append(temp_resp_test, img_response_defect[i, ntrain:(ntrain+ntest)])

print("InD dataset prepared; shape as follows:")
print(temp_img_train.shape)
print(temp_img_test.shape)
print(temp_resp_train.shape)
print(temp_resp_test.shape)
print(Counter(temp_resp_train))
print(Counter(temp_resp_test))
print("Processed InD dataset class distributions:")
temp_resp_train[temp_resp_train == 4] = 3
temp_resp_test[temp_resp_test == 4] = 3
print(Counter(temp_resp_train))
print(Counter(temp_resp_test))

# def relabel_tuples(dsets, ori, target):
#     transformation = dict(zip(ori, target))
#     transformed = []
#     for dpts in tqdm(dsets):
#         transformed.append((dpts[0], transformation[dpts[1]]))
#     return transformed

# Save InD and OOD data
ind_path = os.path.join('..', 'Out-of-Distribution-GANs', 'Datasets', '3DPC')
os.makedirs(ind_path, exist_ok=True)
print('start saving')
ind_train = list(zip(torch.tensor(temp_img_train, dtype=torch.float32).unsqueeze(1), torch.tensor(temp_resp_train, dtype=torch.int64)))
# print(f"Raw Class Distribution: {Counter(list(zip(*ind_train))[1])}")
# ind_train = relabel_tuples(ind_train, [0, 1, 2, 4], [0, 1, 2, 3])  # relabeling to match the task order
# print(f"Processed Class Distribution: {Counter(list(zip(*ind_train))[1])}")
torch.save(ind_train, os.path.join(ind_path, 'ind-train.pt'))
print('InD Training Data Saved')

ind_test = list(zip(torch.tensor(temp_img_test, dtype=torch.float32).unsqueeze(1), torch.tensor(temp_resp_test, dtype=torch.int64)))
# print(f"Raw Class Distribution: {Counter(list(zip(*ind_test))[1])}")
# ind_test = relabel_tuples(ind_test, [0, 1, 2, 4], [0, 1, 2, 3])  # relabeling to match the task order
# print(f"Processed Class Distribution: {Counter(list(zip(*ind_test))[1])}")
torch.save(ind_test, os.path.join(ind_path, 'ind-test.pt'))
print('InD Testing Data Saved')

n_ood_reserved = 2000
n_ood_test = 8000

# OOD Corner Crack
ood_img_train_0 = torch.tensor(img_predictor_defect[2, 0:n_ood_reserved, :, :], dtype=torch.float32).unsqueeze(1)
ood_resp_train_0 = torch.tensor(img_response_defect[2, 0:n_ood_reserved], dtype=torch.int64)
ood_img_test_0 = torch.tensor(img_predictor_defect[2, n_ood_reserved:(n_ood_reserved + n_ood_test), :, :], dtype=torch.float32).unsqueeze(1)
ood_resp_test_0 = torch.tensor(img_response_defect[2, n_ood_reserved:(n_ood_reserved + n_ood_test)], dtype=torch.int64)
print("Corner Crack OOD data prepared; shape as follows:")
print(ood_img_train_0.shape)
print(ood_resp_train_0.shape)
print(ood_img_test_0.shape)
print(ood_resp_test_0.shape)
print(Counter(np.array(ood_resp_train_0)))
print(Counter(np.array(ood_resp_test_0)))

# OOD Long Crack
ood_img_train_1 = torch.tensor(img_predictor_defect[4, 0:n_ood_reserved, :, :], dtype=torch.float32).unsqueeze(1)
ood_resp_train_1 = torch.tensor(img_response_defect[4, 0:n_ood_reserved], dtype=torch.int64)
ood_img_test_1 = torch.tensor(img_predictor_defect[4, n_ood_reserved:(n_ood_reserved + n_ood_test), :, :], dtype=torch.float32).unsqueeze(1)
ood_resp_test_1 = torch.tensor(img_response_defect[4, n_ood_reserved:(n_ood_reserved + n_ood_test)], dtype=torch.int64)
print("Long Crack OOD data prepared; shape as follows:")
print(ood_img_train_1.shape)
print(ood_resp_train_1.shape)
print(ood_img_test_1.shape)
print(ood_resp_test_1.shape)
print(Counter(np.array(ood_resp_train_1)))
print(Counter(np.array(ood_resp_test_1)))

torch.save(list(zip(ood_img_train_0, ood_resp_train_0)), os.path.join(ind_path, 'ood-train-corner-crack.pt'))
torch.save(list(zip(ood_img_train_1, ood_resp_train_1)), os.path.join(ind_path, 'ood-train-long-crack.pt'))
print('Corner Crack OOD Saved')
torch.save(list(zip(ood_img_test_0, ood_resp_test_0)), os.path.join(ind_path, 'ood-test-corner-crack.pt'))
torch.save(list(zip(ood_img_test_1, ood_resp_test_1)), os.path.join(ind_path, 'ood-test-long-crack.pt'))
print('Long Crack OOD Saved')
torch.save(list(zip(ood_img_test_0, ood_resp_test_0)) + list(zip(ood_img_test_1, ood_resp_test_1)), os.path.join(ind_path, 'ood-test.pt'))
print('OOD Testing Data Saved')


# Test
ind_train = torch.load(os.path.join('..', 'Out-of-Distribution-GANs', 'Datasets', '3DPC', 'ind-train.pt'))
ind_val = torch.load(os.path.join('..', 'Out-of-Distribution-GANs', 'Datasets', '3DPC', 'ind-test.pt'))
# print(len(ind_train))

print(ind_train[0][0].shape)
ldr = torch.utils.data.DataLoader(ind_train, 32, True)
class PCDiscriminator(nn.Module):
    def __init__(self):
        super(PCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding = (0, 20))
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 900)
        self.gate = nn.Tanh()

    def forward(self, x):
        out = x.squeeze()
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.gate(self.fc4(out).reshape(-1, 1, 300, 3))
        return out


model = PCDiscriminator()
gmodel = PCGenerator()
for x, y in ldr:
    print(x.shape)
    # x = x.to(dtype=torch.float32)
    # print(Counter(y))
    # out = model(x.unsqueeze(1))
    out = model(x)
    print(out.shape)
    seed = torch.ones((10, 32, 1, 1))
    out = gmodel(seed)
    print(out.shape)
    break