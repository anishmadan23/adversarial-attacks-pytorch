
# coding: utf-8

# In[4]:


import torch
import torchvision


# In[3]:


def get_model(device):
    model = torchvision.models.vgg11(pretrained=True)
    model.to(device)
    model.eval()
    return model

