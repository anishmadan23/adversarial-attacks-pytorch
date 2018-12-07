
# coding: utf-8

# In[9]:


import torch 
import torch.nn as nn
import os
import argparse
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import sys
from PIL import Image
import requests
from io import BytesIO
import urllib.request as url_req
import pickle
from Model import get_model
from utils import *
import json
from model_and_data import Data
from targetedFGSM import FGSM_targeted
from targetedBIM import BIM_targeted
from visualize import visualise
from BIM import BIM


# In[10]:


# %matplotlib inline
# %matplotlib qt


# In[11]:


device  = torch.device('cpu')


# In[12]:


model = get_model(device)                  # loads a pretrained vgg11 model
model.eval()


# In[13]:


def imshow(img,wnid,title=None):
    img = img.cpu().detach().numpy().transpose((1,2,0))
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    
    img = img*std+mean
#     img = np.clip(img,0,1)
    plt.imshow(img)
    
#     title = getClassOfId(wnid)
    
    plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# In[14]:


# url_files = ['valid_urls1.txt','valid_urls2.txt','valid_urls3.txt','valid_urls4.txt','valid_urls5.txt']
# # url_files = ['valid_urls1.txt']
 
# key_to_label_map = pickle.load(open('pickled_id_label_imagenet_map', 'rb')) # load mapping as dict

# for url_file in url_files:
#     with open(url_file) as f:
#         url_dict = json.load(f)
        
#         label_idxs = list(url_dict.keys())
#         print(label_idxs)
#         label = label_idxs[0]
#         key = key_to_label_map[int(label)]['label']
#         print(key,label)
#         url_list = url_dict[label]
#         test_url = url_list[0]
#         test_img = urltoImg(test_url)
        
#         data = Data(model,device, None,None)
#         img_tsor = data.preprocess_data(test_img)
#         img_tsor.unsqueeze_(0)
#         img_tsor = img_tsor.to(device)

#         label = torch.tensor(float(label),requires_grad=False)
#         label = label.to(device)

#         criterion = nn.CrossEntropyLoss()
#         print(img_tsor.size())
#         output = model(img_tsor)
#         _,pred = torch.max(output,1)
#         img_tsor.squeeze_(0)
#         imshow(img_tsor,key)
#         print('Original label = ',float(label))
#         print('Output idx and class =',float(pred.cpu()),',',key_to_label_map[float(pred.cpu())]['label'])


# ### Save all images

# In[15]:


# url_files = ['valid_urls1.txt','valid_urls2.txt','valid_urls3.txt','valid_urls4.txt','valid_urls5.txt']

# for url_file in url_files:
#     with open(url_file) as f:
#         url_dict = json.load(f)
        
#     for label in url_dict.keys():
#         save_imgs(label,url_dict[label])


# In[16]:


imgs = os.listdir('imagenet_imgs/')
all_labels = [img_name.split('_')[0] for img_name in imgs]
unique_labels = np.unique(all_labels)
print(unique_labels)


# In[18]:


imgs = os.listdir('imagenet_imgs/')

# epsilon_arr = list(np.linspace(0,1,21))
# epsilon_arr = [0.01,0.05,0.1,0.5,1]
epsilon_arr = [1.1]
# epsilon_arr = [0.01,0.05]

batch_size = len(os.listdir('imagenet_imgs/'))
fg_top_one_acc_arr = []
fg_top_five_acc_arr = []

fg_t1_local_acc = []             # local is for exact class change
fg_t5_local_acc = []

bim_top_one_acc_arr = []
bim_top_five_acc_arr = []
bim_t1_local_acc = []
bim_t5_local_acc = []

unpert_top_one_acc = []
unpert_top_five_acc = []

top_one_misclassfns = {}
top_five_misclassfns = {}


for epsilon in epsilon_arr:
    top_one_misclassfns['unpert'] = 0
    top_one_misclassfns['fgsm'] = 0
    top_one_misclassfns['fgsm_local'] = 0
    top_one_misclassfns['bim'] = 0
    top_one_misclassfns['bim_local'] = 0
    
    # top_one_misclassfn['llc'] = 0

    top_five_misclassfns['unpert'] = 0
    top_five_misclassfns['fgsm'] = 0
    top_five_misclassfns['fgsm_local'] = 0
    top_five_misclassfns['bim'] = 0
    top_five_misclassfns['bim_local'] = 0
    
    # top_five_misclassfn['llc'] = 0
    
    
    # choose randomly a target class
    targetLabel = unique_labels[np.random.randint(0,len(unique_labels))]
    targetLabel = torch.tensor([int(targetLabel)],requires_grad=False)
    targetLabel = targetLabel.to(device)
    
    for idx,img_name in enumerate(imgs):
            if(idx==105):
                print(idx)
                img_path = os.path.join('imagenet_imgs/',img_name)
                data = Data(model,device, None,None)
                img_tsor = data.preprocess_data(Image.open(img_path))
        #         imshow(img_tsor,'dgs')
                img_tsor.unsqueeze_(0)
                img_tsor = img_tsor.to(device)
                img_tsor.requires_grad_(True)

                label = img_name.split('_')[0]
                label = torch.tensor([int(label)],requires_grad=False)
                label = label.to(device)
        #         print(label.shape)

                criterion = nn.CrossEntropyLoss()
                ############ Unperturbed Model ######################
                unpert_output,unpert_pred, unpert_op_probs, unpert_pred_prob = getPredictionInfo(model,img_tsor)
                unpert_top_probs, unpert_top_labels = predict_top_five(model,img_tsor,k=5)
                
                top_one_misclassfns['unpert'] = checkMatchingLabels(label,unpert_pred,top_one_misclassfns['unpert'])
                top_five_misclassfns['unpert'] = checkMatchingLabelsTop_five(label,unpert_top_labels,top_five_misclassfns['unpert'])
                model.zero_grad()

                class_names = pickle.load(open('pickled_id_label_imagenet_map', 'rb'))
                ############### FGSM ################################
                
                
                # fgsm = FGSM_targeted(model,criterion,img_tsor,label,targetLabel,epsilon)
                # fg_adv_img,fg_perturbation = fgsm.attack()
                
                # fg_output_adv,fg_pred_adv, fg_op_adv_probs, fg_adv_pred_prob  = getPredictionInfo(model,fg_adv_img)
   
                # fg_top_probs,fg_top_labels = predict_top_five(model,fg_adv_img,k=5)

        #         print(int(label),int(pred_adv))
                # top_one_misclassfns['fgsm'] = checkMatchingLabels(label,fg_pred_adv,top_one_misclassfns['fgsm'])
                # top_five_misclassfns['fgsm'] = checkMatchingLabelsTop_five(label,fg_top_labels,top_five_misclassfns['fgsm'])
                
                # top_one_misclassfns['fgsm_local'] = checkMatchingLabels(targetLabel,fg_pred_adv,top_one_misclassfns['fgsm_local'])
                # top_five_misclassfns['fgsm_local'] = checkMatchingLabelsTop_five(targetLabel,fg_top_labels,top_five_misclassfns['fgsm_local'])
                
                model.zero_grad()
                ################ BIM ##############################
                # print('Target Class',class_names[int(targetLabel.numpy())]['label'])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                # bim = BIM_targeted(model,criterion,img_tsor,label,targetLabel,epsilon,epsilon/2,num_iters=0,random_state=False)
                num_iters = 10
                restarts=None
                big_loss_arr = []
                pgd = BIM(model,criterion,img_tsor,label,epsilon,0.01,num_iters=num_iters,random_state=True,restarts=10)
                if restarts is not None:
                    for i in range(restarts):
                        bim_adv_img,bim_perturbation,bim_loss = pgd.attack()
                        big_loss_arr.append(bim_loss)
                        model.zero_grad()
                else:
                     bim_adv_img,bim_perturbation,bim_loss = pgd.attack()

                
                bim_output_adv,bim_pred_adv, bim_op_adv_probs, bim_adv_pred_prob  = getPredictionInfo(model,bim_adv_img)
                
                bim_top_probs,bim_top_labels = predict_top_five(model,bim_adv_img,k=5)

                visualise(img_tsor,bim_perturbation,bim_adv_img,label,label,unpert_pred_prob,bim_pred_adv,bim_adv_pred_prob,epsilon,topkProb=bim_top_probs,topkLabel=bim_top_labels)

                # plt.figure()
                # for i in range(restarts):
                #     plt.plot(big_loss_arr[i])
                # plt.xlabel('Iterations')
                # plt.ylabel('Loss')
                # plt.show()
                # top_one_misclassfns['bim'] = checkMatchingLabels(label,bim_pred_adv,top_one_misclassfns['bim'])
                # top_five_misclassfns['bim'] = checkMatchingLabelsTop_five(label,bim_top_labels,top_five_misclassfns['bim'])
                
                # top_one_misclassfns['bim_local'] = checkMatchingLabels(targetLabel,bim_pred_adv,top_one_misclassfns['bim_local'])
                # top_five_misclassfns['bim_local'] = checkMatchingLabelsTop_five(targetLabel,bim_top_labels,top_five_misclassfns['bim_local'])
                # model.zero_grad()
                
            

                
    # unpert_top_one_acc.append(1-(top_one_misclassfns['unpert']/batch_size))
    # unpert_top_five_acc.append(1-(top_five_misclassfns['unpert']/batch_size))
    # fg_top_one_acc_arr.append(1-(top_one_misclassfns['fgsm']/batch_size))
    # fg_top_five_acc_arr.append(1-(top_five_misclassfns['fgsm']/batch_size))
    # fg_t1_local_acc.append(1-(top_one_misclassfns['fgsm_local']/batch_size))
    # fg_t5_local_acc.append(1-(top_five_misclassfns['fgsm_local']/batch_size))
    # bim_top_one_acc_arr.append(1-(top_one_misclassfns['bim']/batch_size))
    # bim_top_five_acc_arr.append(1-(top_five_misclassfns['bim']/batch_size))
    # bim_t1_local_acc.append(1-(top_one_misclassfns['bim_local']/batch_size))
    # bim_t5_local_acc.append(1-(top_five_misclassfns['bim_local']/batch_size))
    
    # print('Unpert Top 1 Accuracy :',1-(top_one_misclassfns['unpert']/batch_size))
    # print('Unpert Top 5 Accuracy :',1-(top_five_misclassfns['unpert']/batch_size))
    # print('Top 1 FGSM Accuracy :',1-(top_one_misclassfns['fgsm']/batch_size))
    # print('Top 5 FGSM Accuracy :',1-(top_five_misclassfns['fgsm']/batch_size))
    # print('Top 1 FGSM Accuracy(Class Changed) :',1-(top_one_misclassfns['fgsm_local']/batch_size))
    # print('Top 5 FGSM Accuracy(Class Changed) :',1-(top_five_misclassfns['fgsm_local']/batch_size))
    # print('Top 1 BIM Accuracy :',1-(top_one_misclassfns['bim']/batch_size))
    # print('Top 5 BIM Accuracy :',1-(top_five_misclassfns['bim']/batch_size))
    # print('Top 1 BIM Accuracy(Class Changed) :',1-(top_one_misclassfns['bim_local']/batch_size))
    # print('Top 5 BIM Accuracy(Class Changed) :',1-(top_five_misclassfns['bim_local']/batch_size))
    
    


# #         visualise(img_tsor,perturbation,adv_img,label,label,pred_prob,pred_adv,adv_pred_prob,epsilon,topkProb=top_probs,topkLabel=top_labels)


# In[19]:


# epsilon_arr = [0.05]
# plt.figure()
# plt.title('Top-1 Accuracy for FGSM,BIM(Targeted) vs Epsilon')
# plt.plot(epsilon_arr,unpert_top_one_acc, label='Unperturbed Model',marker='o')
# plt.plot(epsilon_arr,fg_top_one_acc_arr,label='FGSM',marker='o')
# plt.plot(epsilon_arr,bim_top_one_acc_arr,label='BIM',marker='o')
# plt.plot(epsilon_arr,fg_t1_local_acc,label='FGSM(changed_to_target)',marker='o')
# plt.plot(epsilon_arr,bim_t1_local_acc,label='BIM(changed_to_target)',marker='o')
# plt.ylabel('Top-1 Accuracy')
# plt.xlabel('Epsilon')
# plt.legend()


# plt.figure()
# plt.title('Top-5 Accuracy for FGSM,BIM(Targeted) vs Epsilon')
# plt.plot(epsilon_arr,unpert_top_five_acc,label='Unperturbed Model',marker='o')
# plt.plot(epsilon_arr,fg_top_five_acc_arr,label='FGSM',marker='o')
# plt.plot(epsilon_arr,bim_top_five_acc_arr,label='BIM',marker='o')
# plt.plot(epsilon_arr,fg_t5_local_acc,label='FGSM(changed_to_target)',marker='o')
# plt.plot(epsilon_arr,bim_t5_local_acc,label='BIM(changed_to_target)',marker='o')
# plt.ylabel('Top-5 Accuracy')
# plt.xlabel('Epsilon')
# plt.legend()
# plt.show()
# #         visualise(img_tsor,perturbation,adv_img,label,label,pred_prob,pred_adv,adv_pred_prob,epsilon,topkProb=top_probs,topkLabel=top_labels)
# print('Epsilon',epsilon_arr)
# print('Unperturbed')
# print(unpert_top_one_acc)
# print(unpert_top_five_acc)
# print('FGSM')
# print(fg_top_one_acc_arr)
# print(fg_top_five_acc_arr)
# print('BIM')
# print(bim_top_one_acc_arr)
# print(bim_top_five_acc_arr)
# print('FGSM_local')
# print(fg_t1_local_acc)
# print(fg_t5_local_acc)
# print('BIM_local')
# print(bim_t1_local_acc)
# print(bim_t5_local_acc)
# print('DONE DONA DONE')

