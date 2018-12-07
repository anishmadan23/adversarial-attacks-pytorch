
# coding: utf-8

# In[2]:


import torchvision
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import pickle
from PIL import Image
import requests
from io import BytesIO
import urllib.request as url_req
from utils import urltoImg
import json


# In[3]:


def get_model(device):
    model = torchvision.models.vgg11(pretrained=True)
    model.to(device)
    model.eval()
    return model


# In[12]:


class Data(object):
    def __init__(self,model,device,testClassList,key):
        self.model = model
        self.device = device
        self.testClassList = testClassList
        self.id_label_map = pickle.load(open('pickled_id_label_imagenet_map', 'rb')) # load mapping as dict
        self.file_key = key
        
    def query_class_data(self,idx): 
        
        idx_id = self.id_label_map[idx]['id']
        wnid = str('n')+str(idx_id.split('-')[0])
        # print(wnid)
        synset_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='+str(wnid)
        # print(synset_url)
        urls = requests.get(synset_url).text.split('\r\n')     #clean for valid url
        label_class = self.id_label_map[idx]['label']               # get class name for given idx
        # print(label_class)
        return urls
        
    
    def preprocess_data(self,img):
#         for idx in self.testClassList:
#             urls = query_class_data(idx)
#             print(idx,len(urls))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize])
        
        return preprocess(img)
    
    def testModel(self,img,label):
        img_tsor = self.preprocess_data(img)
        img_tsor.unsqueeze_(0)
        img_tsor = img_tsor.to(self.device)

        label = torch.tensor(label,requires_grad=False)
        label = label.to(self.device)

        criterion = nn.CrossEntropyLoss()
        print(img_tsor.size())
        output = self.model(img_tsor)
        _,pred = torch.max(output,1)
        img_tsor.squeeze_(0)
#         imshow(img_tsor,wnid)
        org_label = float(label)
        output_idx = float(pred.cpu())
        output_class = self.id_label_map[float(pred.cpu())]['label']
        print('Original label = ',org_label)
        print('Output idx and class =',output_idx,',',output_class)
        return org_label,output_idx
    
            
    def make_valid_url_list(self):
        valid_urls = {}
        for idx in self.testClassList:
            urls = self.query_class_data(idx)
            np.random.shuffle(urls)
            i = 0
            valid_urls[idx] = []
            img = None
            flag = 0
            while (i<len(urls)):
                if (len(valid_urls[idx])==5):
                    break
                try:
                    var = url_req.urlopen(urls[i])
                    redirected_url = var.geturl()
                    if (str(redirected_url.split('/')[-1].split('.')[0]) == 'photo_unavailable'):   # this always results in 'photo_unavailable' if photo no longer exists
                        print(urls[i])
                        print('No photo')
                    else:
                        img = urltoImg(urls[i])
                        flag=1
                except:
                    print(urls[i])
                    print('Invalid url')
                
                if(flag==1):
                    orig_label,pred_label = self.testModel(img,idx)
                    if(orig_label != pred_label):
                        print('Diff label',i)
    
                    else:
                        print('Valid_url',i)
                        valid_urls[idx].append(urls[i])
                flag=0 
                i+=1
            file_name = str('valid_urls')+str(key)+str('.txt')
            with open(file_name,'a+') as f:
                f.write(json.dumps(valid_urls))
        return valid_urls
    


# In[20]:


# class_id_arr1 = [8,9,21,31,35,49,63,75,84,86]
# class_id_arr2 = [93,100,105,113,121,130,144,148,151,282]
# class_id_arr3 = [293,295,298,309,311,314,360,417,430]
# class_id_arr4 = [438,457,470,480,491,543,546,568,578,587,609]
# class_id_arr5 = [620,629,637,668,696,706,721,773,806,845]

# len(class_id_arr)


# In[21]:


# device = torch.device('cuda')

# model = get_model(device)


# In[22]:


# key = 4
# dset = Data(model,device,class_id_arr4,key)
# val_urls = dset.make_valid_url_list()


# In[23]:


# key = 5
# dset = Data(model,device,class_id_arr5,key)
# val_urls = dset.make_valid_url_list()

