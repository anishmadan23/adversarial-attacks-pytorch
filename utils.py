
# coding: utf-8

# In[2]:


import requests
from io import BytesIO
import urllib.request as url_req
from PIL import Image
import os
import torch
import torch.nn.functional as F


# In[3]:


def urltoImg(url):
    print(url)
    try:
        img = Image.open(url_req.urlopen(url))
    except Exception as error:
        print("Couldn't load image "+str(error))
    
    return img 


# In[1]:


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x

def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)

def torch_arctanh(x, eps=1e-6):
    x = x*(1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5

def save_imgs(key,url_list):
    for i,url in enumerate(url_list):
        img = urltoImg(url)
        save_path = os.path.join('imagenet_imgs',str(key)+str('_')+str(i)+str('.png'))
        img.save(save_path)

def predict_top_five(model,img,k=5):

    output = model(img)
    # print(output.size())
    op_probs = F.softmax(output,dim=1)
    top_k = torch.topk(output,k,dim=1)
    labels = top_k[1].squeeze_(0)
    labels_np = labels.cpu().numpy()
    # print(labels)
    op_probs_np = op_probs.squeeze_(0).detach().cpu().numpy()*100
    # print('Probs')
    # print(op_probs_np[labels_np])



    return op_probs_np[labels_np],labels_np

def getPredictionInfo(model,img):
    output = model(torch.tensor(img))
    _,pred = torch.max(output.data,1)
    #     print(adv_img.data-img.data)
    op_probs = F.softmax(output, dim=1)                 #get probability distribution over classes
    pred_prob =  ((torch.max(op_probs.data, 1)[0][0]) * 100, 4)      #find probability (confidence) of a predicted class
    return output,pred,op_probs,pred_prob

def checkMatchingLabels(label,pred_label,misclassfns):
    if (int(label)!=int(pred_label)):
        misclassfns+=1
    return misclassfns

def checkMatchingLabelsTop_five(label,pred_label_list,misclassfns):
    if(int(label) not in pred_label_list.astype(int)):
        misclassfns+=1
    return misclassfns