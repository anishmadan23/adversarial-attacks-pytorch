
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from Model import get_model
from utils import *
from visualize import visualise
import math


# In[ ]:


class BIM_targeted(object):
    def __init__(self,model,criterion,orig_img,orig_label,targetLabel,eps,alpha,num_iters=0,random_state=False):
        self.model = model
        self.criterion = criterion
        self.orig_img = orig_img.clone()
        self.eps = eps
        self.orig_label = orig_label
        self.targetLabel = targetLabel
        self.alpha = alpha
        self.rand = random_state
        self.img_bim = torch.tensor(orig_img.data,requires_grad=True)
        if not random_state:
            self.num_iters = math.ceil(min(self.eps+4,1.25*self.eps))
        else:
            self.num_iters=num_iters
        # self.num_iters = 3
    def attack(self):
        if self.rand:                   # attack changes from BIM to Madry's PGD
            delta_init = torch.from_numpy(np.random.uniform(-self.eps,self.eps,self.orig_img.shape)).type(torch.FloatTensor)
            self.img_bim = torch.tensor(self.img_bim.data+ delta_init,requires_grad=True)
            clipped_delta = torch.clamp(self.img_bim.data-self.orig_img.data, -self.eps,self.eps)
            self.img_bim = torch.tensor(self.orig_img.data+clipped_delta,requires_grad=True)



        loss_arr = []
        
        output_tr,pred_label,op_probs,pred_prob = getPredictionInfo(self.model,self.orig_img)
        # output_tr = self.model(self.orig_img)
        # op_probs = F.softmax(output_tr,dim=1)
        # pred_prob = ((torch.max(op_probs.data, 1)[0][0]) * 100, 4)
        # _,pred_label = torch.max(output_tr.data,1)
        
        # iterative attack
#         print('Iters',self.num_iters)
        for i in range(self.num_iters):
#             print(i)
            output = self.model(self.img_bim)
#             print(type(output))
#             print(type(self.label))
            loss = self.criterion(output, self.targetLabel)        # compute loss be output and target label
#             print(loss)
            loss.backward()
            delta = self.alpha * torch.sign(self.img_bim.grad.data)
            self.img_bim = torch.tensor(self.img_bim.data - delta, requires_grad=True) # adversary without clipping
            
            clipped_delta = torch.clamp(self.img_bim.data-self.orig_img.data, -self.eps,self.eps) #clipping the delta
            self.img_bim = torch.tensor(self.orig_img.data-clipped_delta,requires_grad=True) # adding the clipped delta to original image
            loss_arr.append(loss)
        return self.img_bim, clipped_delta, loss_arr

