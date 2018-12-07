
# coding: utf-8

# In[3]:


import torch
import torch.nn.functional as F
from utils import *

# In[4]:


class FGSM(object):
    def __init__(self,model,criterion,orig_img,orig_label,eps):
        self.model = model
        self.criterion = criterion                                                                          
        self.orig_img = orig_img
        self.epsilon = eps
        self.orig_label = orig_label
        
    def attack(self):
        # output = self.model(self.orig_img)
        # op_probs = F.softmax(output,dim=1)
        # pred_prob = ((torch.max(op_probs.data, 1)[0][0]) * 100, 4)
        # _,pred_label = torch.max(output.data,1)
        output,pred,op_probs,pred_prob = getPredictionInfo(self.model,self.orig_img)

        # print(output.size())
        # print(self.orig_label)
        loss = self.criterion(output,self.orig_label)
        # print(loss)
        loss.backward()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        img_grad = torch.sign(self.orig_img.grad.data)             # sign of the gradient
        adv_img = self.orig_img.data + self.epsilon*img_grad
#         output_adv = self.model(torch.tensor(adv_img))
#         _,pred_adv = torch.max(output_adv.data,1)
#     #     print(adv_img.data-img.data)
#         op_adv_probs = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
#         adv_pred_prob =  ((torch.max(op_adv_probs.data, 1)[0][0]) * 100, 4)      #find probability (confidence) of a predicted class
    #     print(float(adv_pred_prob[0]),float(pred_adv))

        return adv_img,img_grad

