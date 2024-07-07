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

from copy import deepcopy

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
for defect_type in range(n_defect):
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

# sort the last column
for i in range(len(img_predictor_normal)):
    img_predictor_normal[i, :, :] = np.array(sorted(img_predictor_normal[i, :, :], key = operator.itemgetter(2, 1)))

#% organize dataset as multiple stages
ntrain = 8000
ntest = 2000

# task lists
# 1 small dent, 2 corner crack, 3 big dent, 4 wood grain, 5 long crack
task_init = np.array([0, 1])
task_ood = np.array([2])
task_seq = np.array([3, 4])

train_data = {}
test_data = {}
multitask_train_loader = {}
multitask_test_loader = {}

# baseline dataset
temp_img_train = img_predictor_normal[0:ntrain, :, :]
temp_resp_train = img_response_normal[0:ntrain]
temp_img_test = img_predictor_normal[ntrain:(ntrain+ntest), :, :]
temp_resp_test = img_response_normal[ntrain:(ntrain+ntest)]
for i in task_init:
    temp_img_train = np.vstack((temp_img_train, img_predictor_defect[i, 0:ntrain, :, :]))
    temp_resp_train = np.append(temp_resp_train, img_response_defect[i, 0:ntrain])
    temp_img_test = np.vstack((temp_img_test, img_predictor_defect[i, ntrain:(ntrain+ntest), :, :]))
    temp_resp_test = np.append(temp_resp_test, img_response_defect[i, ntrain:(ntrain+ntest)])
    
train_data[0] = TensorDataset(torch.tensor(temp_img_train).unsqueeze(1), torch.tensor(temp_resp_train))
test_data[0] = TensorDataset(torch.tensor(temp_img_test).unsqueeze(1), torch.tensor(temp_resp_test))
multitask_train_loader[0] = torch.utils.data.DataLoader(train_data[0], batch_size = 10, shuffle = True)
multitask_test_loader[0] = torch.utils.data.DataLoader(test_data[0], batch_size = 10, shuffle = True)

# sequential multistage dataset
task_ind = 1
for task in task_seq:
    train_data[task_ind] = TensorDataset(torch.tensor(img_predictor_defect[task, 0:ntrain, :, :]).unsqueeze(1), torch.tensor(img_response_defect[task, 0:ntrain]))
    test_data[task_ind] = TensorDataset(torch.tensor(img_predictor_defect[task, ntrain:(ntrain+ntest), :, :]).unsqueeze(1), torch.tensor(img_response_defect[task, ntrain:(ntrain+ntest)]))
    multitask_train_loader[task_ind] = torch.utils.data.DataLoader(train_data[task_ind], batch_size = 10, shuffle = True)
    multitask_test_loader[task_ind] = torch.utils.data.DataLoader(test_data[task_ind], batch_size = 10, shuffle = True)
    task_ind += 1

# adversarial dataset in-distribution
temp_img_train = img_predictor_normal[ntrain:(ntrain+int(ntest/2)), :, :]
temp_resp_train = img_response_normal[ntrain:(ntrain+int(ntest/2))]
temp_img_test = img_predictor_normal[(ntrain+int(ntest/2)):(ntrain+ntest), :, :]
temp_resp_test = img_response_normal[(ntrain+int(ntest/2)):(ntrain+ntest)]
for i in task_init:
    temp_img_train = np.vstack((temp_img_train, img_predictor_defect[i, ntrain:(ntrain+int(ntest/2)), :, :]))
    temp_resp_train = np.append(temp_resp_train, img_response_defect[i, ntrain:(ntrain+int(ntest/2))])
    temp_img_test = np.vstack((temp_img_test, img_predictor_defect[i, (ntrain+int(ntest/2)):(ntrain+ntest), :, :]))
    temp_resp_test = np.append(temp_resp_test, img_response_defect[i, (ntrain+int(ntest/2)):(ntrain+ntest)])
    
ind_dataset_val_for_train = TensorDataset(torch.tensor(temp_img_train).unsqueeze(1), torch.tensor(temp_resp_train))
ind_dataset_val_for_test = TensorDataset(torch.tensor(temp_img_test).unsqueeze(1), torch.tensor(temp_resp_test))
ind_dataloader_val_for_train = torch.utils.data.DataLoader(ind_dataset_val_for_train, batch_size = 10, shuffle = True)
ind_dataloader_val_for_test = torch.utils.data.DataLoader(ind_dataset_val_for_test, batch_size = 10, shuffle = True)

# adversarial dataset out-of-distribution
temp_img_train = np.zeros((1, size_img, 3))
temp_resp_train = np.zeros(1)
temp_img_test = np.zeros((1, size_img, 3))
temp_resp_test = np.zeros(1)
for i in task_ood:
    temp_img_train = np.vstack((temp_img_train, img_predictor_defect[i, 0:ntrain, :, :]))
    temp_resp_train = np.append(temp_resp_train, img_response_defect[i, 0:ntrain])
    temp_img_test = np.vstack((temp_img_test, img_predictor_defect[i, ntrain:(ntrain+ntest), :, :]))
    temp_resp_test = np.append(temp_resp_test, img_response_defect[i, ntrain:(ntrain+ntest)])
    
temp_img_train = np.delete(temp_img_train, (0), axis = 0)
temp_resp_train = np.delete(temp_resp_train, (0), axis = 0) 
temp_img_test = np.delete(temp_img_test, (0), axis = 0)
temp_resp_test = np.delete(temp_resp_test, (0), axis = 0) 

ood_dataset_val_for_train = TensorDataset(torch.tensor(temp_img_train).unsqueeze(1), torch.tensor(temp_resp_train))
ood_dataset_val_for_test = TensorDataset(torch.tensor(temp_img_test).unsqueeze(1), torch.tensor(temp_resp_test))
ood_dataloader_val_for_train = torch.utils.data.DataLoader(ood_dataset_val_for_train, batch_size = 10, shuffle = True)
ood_dataloader_val_for_test = torch.utils.data.DataLoader(ood_dataset_val_for_test, batch_size = 10, shuffle = True)


#% out-of-distribution setup
# network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding = (0, 20))
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#from architecture_res_pt import ResNet

net = Net()
net.cuda()

# store confusion matrix
conf_mat = {}
conf_count = 0

# accuracy matrix
acc = {}
acc[0] = []

# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.0005, momentum = 0.9)
nepochs = 200

# train baseline model:
for epoch in range(nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(multitask_train_loader[0], 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = variable(inputs.float()), variable(labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # test every iteration
    acc[0].append(test(net, multitask_test_loader[0]))

print('Finished Training')

# save baseline model
PATH = "baseline_model.pt"
torch.save(net.state_dict(), PATH)

#% test the neural network
correct = 0
total = 0
input_labels = np.zeros(1)
pred_labels = np.zeros(1)

with torch.no_grad():
    for data in multitask_test_loader[0]:
        images, labels = data
        images, labels = variable(images.float()), variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        input_labels = np.append(input_labels, labels.cpu())
        pred_labels = np.append(pred_labels, predicted.cpu())
        
input_labels = np.delete(input_labels, (0), axis = 0) 
pred_labels = np.delete(pred_labels, (0), axis = 0) 
conf_matrix = confusion_matrix(pred_labels, input_labels)
print(conf_matrix)
        
print('Accuracy of the network on the 14000 test images: %d %%' % (100 * correct / total))


#% train ODIN scores
std = (0.381,)
logger.info("search ODIN params")
best_temperature, best_magnitude = search_ODIN_hyperparams(net, ind_dataloader_val_for_train, ood_dataloader_val_for_train, ind_dataloader_val_for_test, ood_dataloader_val_for_test, std=std)
ind_scores_val_for_train = get_ODIN_score(net, ind_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ood_scores_val_for_train = get_ODIN_score(net, ood_dataloader_val_for_train, best_magnitude, best_temperature, std=std)
ind_features_val_for_train = ind_scores_val_for_train.reshape(-1,1)
ood_features_val_for_train = ood_scores_val_for_train.reshape(-1,1)

# ----- Train OoD detector using validation data -----
from scoreslib.metric import get_metrics, train_lr
lr = train_lr(ind_features_val_for_train, ood_features_val_for_train)
# ----- Calculating metrics using test data -----
ind_scores_val_for_test = get_ODIN_score(net, ind_dataloader_val_for_test, best_magnitude, best_temperature, std=std)
ood_scores_val_for_test = get_ODIN_score(net, ood_dataloader_val_for_test, best_magnitude, best_temperature, std=std)
ind_features_val_for_test = ind_scores_val_for_test.reshape(-1,1)
ood_features_val_for_test = ood_scores_val_for_test.reshape(-1,1)
metrics = get_metrics(lr, ind_features_val_for_test, ood_features_val_for_test, acc_type="best")
print(metrics)

# detect new Task 1
threshold = np.quantile(ind_features_val_for_train, 0.2)
print((np.sum(ind_features_val_for_train >= threshold), ind_features_val_for_train.shape[0]))
print((np.sum(ood_features_val_for_train < threshold), ood_features_val_for_train.shape[0]))
#% check ood datasets
for i in range(len(task_seq)):
    incoming_score = get_ODIN_score(net, multitask_train_loader[i + 1], best_magnitude, best_temperature, std=std)
    incoming_features_test = incoming_score.reshape(-1, 1)
    print((np.sum(incoming_features_test < threshold), incoming_features_test.shape[0]))
    
#%
incoming_score_1 = get_ODIN_score(net, multitask_train_loader[1], best_magnitude, best_temperature, std=std)
incoming_features_test_1 = incoming_score_1.reshape(-1, 1)
incoming_score_2 = get_ODIN_score(net, multitask_train_loader[2], best_magnitude, best_temperature, std=std)
incoming_features_test_2 = incoming_score_2.reshape(-1, 1)

fig, axs = plt.subplots(4, 1, sharey=True, tight_layout=True)
n_bins = 100
axs[0].hist(ind_features_val_for_train, range = [0, 1], bins=n_bins)
axs[1].hist(ood_features_val_for_train, range = [0, 1], bins=n_bins)
axs[2].hist(incoming_features_test_1, range = [0, 1], bins=n_bins)
axs[3].hist(incoming_features_test_2, range = [0, 1], bins=n_bins)

#% sub-functions
net = Net()
net.load_state_dict(torch.load(PATH))

def sampleinspection(old_data, new_data, ood_parameters, threshold):
    global conf_count
    combined_data = ConcatDataset((old_data, new_data))
    new_loader = torch.utils.data.DataLoader(combined_data, batch_size = 10, shuffle = False, num_workers = 0)

    incoming_score = get_ODIN_score(net, new_loader, ood_parameters[0], ood_parameters[1], std=std)
    incoming_features_test = incoming_score.reshape(-1, 1)
    res_ood_detector = (incoming_features_test < threshold)
    
    res_ood_true = np.append(np.repeat(False, len(old_data)), np.repeat(True, len(new_data)))
    print(confusion_matrix(res_ood_detector, res_ood_true))

    conf_mat[conf_count] = confusion_matrix(res_ood_detector, res_ood_true)
    conf_count += 1
    
    out_inputs = np.zeros((1, 1, size_img, 3))
    out_labels = np.zeros(1)   
    k = 0
    for i in range(len(old_data)):
        #if incoming_features_test[k] <= threshold:
        out_labels = np.append(out_labels, old_data[i][1].numpy())
        out_inputs = np.vstack((out_inputs, np.expand_dims(old_data[i][0].numpy(), axis=0)))            
        k += 1
        
    for i in range(len(new_data)):
        if incoming_features_test[k] < threshold:
            out_labels = np.append(out_labels, new_data[i][1].numpy())
            out_inputs = np.vstack((out_inputs, np.expand_dims(new_data[i][0].numpy(), axis=0)))           
        k += 1

    # delete first element
    out_inputs = np.delete(out_inputs, (0), axis = 0)
    out_labels = np.delete(out_labels, (0), axis = 0) 
    out_data = TensorDataset(torch.tensor(out_inputs).float(), torch.tensor(out_labels).float())
    label_set_post = label_set_prev | set(np.unique(out_labels).astype(int).tolist())
    
    return([torch.utils.data.DataLoader(out_data, batch_size = 10, shuffle = True, num_workers = 0), label_set_post])
    
ood_parameters = [best_magnitude, best_temperature]

#% start out-of-distribution learning
# threshold of softmax for out-of-distribution detection
old_sample_size = 3000
importance = 0.1
lambda_2 = 0.1
# load baseline model
net = Net()
net.load_state_dict(torch.load(PATH))

#nepochs = 20
loss = {}
label_set_prev = set(task_init)
subtask_test_loader = {}
ntasks = len(task_seq)

# record dataset in the previous step
data_loader_prev = {}
data_loader_prev[0] = multitask_train_loader[0]

#incremental_l = incremental_Loss.apply

for i in range(ntasks):
    task = i + 1
    
    old_tasks = []
    old_images = np.zeros((1, 1, size_img, 3))
    old_labels = np.zeros(1)
    for sub_task in range(task):
        temp_data = data_loader_prev[sub_task].dataset
        # use full dataset if the sample size if smaller than desired
        temp_sample_size = old_sample_size
        if len(temp_data) < temp_sample_size:
            temp_sample_size = len(temp_data)
            
        # sample previous samples for training
        sample_idx = random.sample(range(len(temp_data)), temp_sample_size)
        for ii in sample_idx:
            old_tasks = old_tasks + [temp_data[ii][0]]
            old_images = np.vstack((old_images, np.expand_dims(temp_data[ii][0], axis = 0)))
            old_labels = np.append(old_labels, temp_data[ii][1])
    old_tasks = random.sample(old_tasks, k=temp_sample_size)
    old_images = np.delete(old_images, (0), axis = 0)
    old_labels = np.delete(old_labels, (0), axis = 0) 
    old_data = TensorDataset(torch.tensor(old_images).float(), torch.tensor(old_labels).float())

    dataloader_iterator = iter(multitask_train_loader[task])
    new_images = np.zeros((1, 1, size_img, 3))
    new_labels = np.zeros(1)
    for j in range(len(multitask_train_loader[task])):
        inputs, labels = next(dataloader_iterator)
        new_images = np.vstack((new_images, inputs.numpy()))
        new_labels = np.append(new_labels, labels)
    new_images = np.delete(new_images, (0), axis = 0)
    new_labels = np.delete(new_labels, (0), axis = 0)         
    new_data = TensorDataset(torch.tensor(new_images).float(), torch.tensor(new_labels).float())    
    
    loss[task] = []
    acc[task] = []
    ood_loader, label_set_post = sampleinspection(old_data, new_data, ood_parameters, threshold)
    data_loader_prev[task] = ood_loader

    net = Net()
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum = 0.9)
    
    
    for _ in tqdm(range(nepochs)):
        ewc = EWC(net, old_tasks)
        net.train()
        epoch_loss = 0
        for (InD_images, InD_labels), (OOD_images, OOD_labels) in zip(ood_loader, ood_dataloader_val_for_train):
            optimizer.zero_grad()
            
            InD_images, InD_labels = variable(InD_images), variable(InD_labels)
            OOD_images, OOD_labels = variable(OOD_images), variable(OOD_labels)
            
            
            InD_pred = net(InD_images.float())
            ood_temp_loader = torch.utils.data.DataLoader(TensorDataset(OOD_images, OOD_labels), batch_size = 10, shuffle = True, num_workers = 0)
            ind_temp_loader = torch.utils.data.DataLoader(TensorDataset(InD_images, InD_labels), batch_size = 10, shuffle = True, num_workers = 0)
            
            ood_scores = get_ODIN_score(net, ood_temp_loader, best_magnitude, best_temperature, std=std).reshape(-1, 1)
            ind_scores = get_ODIN_score(net, ind_temp_loader, best_magnitude, best_temperature, std=std).reshape(-1, 1)
            
            if len(ind_scores) < 10:
                ind_scores = np.concatenate((ind_scores, np.zeros((10 - len(ind_scores), 1))))
            if len(ood_scores) < 10:
                ood_scores = np.concatenate((ood_scores, np.zeros((10 - len(ood_scores), 1))))                
            
            penalty_ood = np.sum(np.maximum(np.zeros(ood_scores.shape), ood_scores - np.tile(threshold, ind_scores.shape)))
            penalty_ind = np.sum(np.maximum(np.zeros(ind_scores.shape), np.tile(threshold, ind_scores.shape) - ind_scores))
            loss_temp = F.cross_entropy(InD_pred, InD_labels.long()) + importance * ewc.penalty(net) + lambda_2 * (penalty_ind + penalty_ood)
            epoch_loss += loss_temp.data
            loss_temp.backward()
            optimizer.step()
        
        loss[task].append(epoch_loss / len(ood_loader))
        
        for sub_task in range(task + 1):
            acc[sub_task].append(test(net, multitask_test_loader[sub_task]))
    
    # update ood parameters
#    logger.info("search ODIN params")
#    best_temperature, best_magnitude = search_ODIN_hyperparams(net, ind_dataloader_val_for_train, ood_dataloader_val_for_train, ind_dataloader_val_for_test, ood_dataloader_val_for_test, std=std)
#    ood_parameters = [best_magnitude, best_temperature]
    
    # update threshold
    ind_scores_val_for_train = get_ODIN_score(net, ood_loader, best_magnitude, best_temperature, std=std)
    ind_features_val_for_train = ind_scores_val_for_train.reshape(-1,1)
    threshold = np.quantile(ind_features_val_for_train, 0.2)
    
    # update set
    label_set_prev = label_set_post

#%% plots
def accuracy_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * nepochs, (ntasks + 1) * nepochs)), v)
    plt.ylim(0, 1)
accuracy_plot(acc)

#%%
l0 = plt.plot(list(range(0 * nepochs, (ntasks + 1) * nepochs)), acc[0], '-', linewidth = 1.5, label = "normal, small dent, corner crack")
l1 = plt.plot(list(range(1 * nepochs, (ntasks + 1) * nepochs)), acc[1], '--', linewidth = 1.5, label = "wood grain")
l2 = plt.plot(list(range(2 * nepochs, (ntasks + 1) * nepochs)), acc[2], '-.', linewidth = 1.5, label = "long crack")
plt.legend(loc = 'lower left', fontsize = 12)
plt.xlabel('epochs', fontsize = 18)
plt.ylabel('test accuracy', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.ylim(0, 1)

#%%  test the final neural network
multitask_test_loader = {}
multitask_test_loader[0] = torch.utils.data.DataLoader(test_data[0], batch_size = 10, shuffle = False)
# sequential multistage dataset
task_ind = 1
for task in task_seq:
    multitask_test_loader[task_ind] = torch.utils.data.DataLoader(test_data[task_ind], batch_size = 10, shuffle = False)
    task_ind += 1
    
correct = 0
total = 0
input_labels = np.zeros(1)
pred_labels = np.zeros(1)

with torch.no_grad():
    for j in range(ntasks + 1):
        for data in multitask_test_loader[j]:
            images, labels = data
            images, labels = variable(images.float()), variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            input_labels = np.append(input_labels, labels.cpu())
            pred_labels = np.append(pred_labels, predicted.cpu())
        
input_labels = np.delete(input_labels, (0), axis = 0) 
pred_labels = np.delete(pred_labels, (0), axis = 0) 
conf_matrix = confusion_matrix(pred_labels, input_labels)
print(conf_matrix)
        
print('Accuracy of the network on the 14000 test images: %d %%' % (100 * correct / total))


#%% test by classes
#%%  test the final neural network
sample_incorrect = np.zeros((ntasks + 1, ntest))
predictor_incorrect = np.zeros((ntasks + 1, ntest, size_img, 3))
label_incorrect = np.zeros((ntasks + 1, ntest))
pred_incorrect = np.zeros((ntasks + 1, ntest))
with torch.no_grad():
    for j in range(ntasks + 1):
        count = -1
        count_incorrect = 0
        for data in multitask_test_loader[j]:
            images, labels = data
            images, labels = variable(images.float()), variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            
            for k in range(c.cpu().numpy().shape[0]):
                
                if predicted[k] != labels[k]:
                    sample_incorrect[j, count_incorrect] = count + k + 1
                    if j == 0:
                        if count + k + 1 < ntest:
                            image_center_temp = img_center_normal[ntrain + count + k + 1, :]
                        else:
                            if count + k + 1 < 2 * ntest:
                                image_center_temp = img_center_defect[0, ntrain + count + k + 1 - ntest, :]    
                            if count + k + 1 > 2 * ntest:
                                image_center_temp = img_center_defect[1, ntrain + count + k + 1 - 2 * ntest, :]
                    
                    if j > 0:
                        image_center_temp = img_center_defect[j + 2, ntrain + count + k + 1, :]
                    
                    predictor_incorrect[j, count_incorrect, :, :] = images[k, :, :, :].cpu().numpy().squeeze() + np.tile(image_center_temp, (size_img, 1))
                    label_incorrect[j, count_incorrect] = labels[k].cpu().numpy()
                    pred_incorrect[j, count_incorrect] = predicted[k].cpu().numpy()
                    count_incorrect += 1
                    
                    if j == 0:
                        print('Misclassified Sample %5s (originally in Class %1s) to Class %1s' % (count, labels[k].cpu().numpy(), predicted[k].cpu().numpy()))
            
            count += 10


#%% visualization as 3D point clouds
color_map = np.array([[1, 1, 1], [1, 0.5, 0], [0.5, 1, 0], [0, 0, 1], [0.5, 1, 1], [1, 0, 0]])
            
# view all samples
v = pptk.viewer(np.concatenate((indat_defect_raw[:, 0 : 3], indat_normal_raw[:, 0 : 3]), axis = 0), np.concatenate((indat_defect_raw[:, 3], indat_normal_raw[:, 3])))
v.set(point_size = 0.1)
v.color_map(color_map, scale = [0, 5])

# view centers of training samples
v = pptk.viewer(np.concatenate((img_center_defect[:, 0 : ntrain, :].reshape((1, 5*ntrain, 3)).squeeze(), img_center_normal[0 : ntrain, :]), axis = 0), np.concatenate((img_response_defect[:, 0 : ntrain].reshape((1, 5*ntrain)).squeeze(), img_response_normal[0 : ntrain])))
v.set(point_size = 0.3)
v.color_map(color_map, scale = [0, 5])

# view centers of test samples
v = pptk.viewer(np.concatenate((img_center_defect[:, ntrain : (ntrain+ntest), :].reshape((1, 5*ntest, 3)).squeeze(), img_center_normal[ntrain : (ntrain+ntest), :]), axis = 0), np.concatenate((img_response_defect[:, ntrain : (ntrain+ntest)].reshape((1, 5*ntest)).squeeze(), img_response_normal[ntrain : (ntrain+ntest)])))
v.set(point_size = 1)
v.color_map(color_map)

#%% view individual mis-classified samples
color_map_mis = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1], [1, 0, 0]])
task_ind = 2
incorrect_ind = 10
resp_temp = np.concatenate((np.ones(len(indat_defect_raw)), np.zeros(len(indat_normal_raw)), np.ones(size_img) * 2))
v = pptk.viewer(np.concatenate((indat_defect_raw[:, 0 : 3], indat_normal_raw[:, 0 : 3], predictor_incorrect[task_ind, incorrect_ind, :, :]), axis = 0), resp_temp)
v.set(point_size = 0.1)
v.color_map(color_map_mis, scale = [0, 2])

print(label_incorrect[task_ind, incorrect_ind])
print (pred_incorrect[task_ind, incorrect_ind])

# generate the heatmaps
#default_cmap = LinearSegmentedColormap.from_list('custom blue', 
#                                                 [(0, '#ffffff'),
#                                                  (0.25, '#000000'),
#                                                  (1, '#000000')], N=256)  
