import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pylab import savefig



def visualise(img,img_grad,img_adv,true_label,pred_unpert_label,unpert_pred_prob,adv_label,adv_prob,epsilon,topkProb=None,topkLabel=None):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = img.cpu()
    img = img.squeeze(0)
    img = img.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()#reverse of normalization op- "unnormalize"
    img = np.transpose( img , (1,2,0))   # C X H X W  ==>   H X W X C
    img = np.clip(img, 0, 1)    
    
    img_adv = img_adv.cpu()
    img_adv = img_adv.squeeze(0)
    img_adv = img_adv.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).detach().numpy()#reverse of normalization op- "unnormalize"
    img_adv = np.transpose( img_adv, (1,2,0))   # C X H X W  ==>   H X W X C
    img_adv = np.clip(img_adv, 0, 1)
    
    if img_grad is  not None:
        img_grad= img_grad.cpu().squeeze(0).detach().numpy()
        img_grad = np.transpose(img_grad, (1,2,0))
        img_grad = np.clip(img_grad, 0, 1)
    
        
        figure, ax = plt.subplots(1,3, figsize=(10,10))
        plt.subplots_adjust(wspace=0.4)
        
        
    #     print(img.shape)
        ax[0].imshow(img)
        ax[0].set_title('Clean Image', fontsize=20)
        
        
        
        ax[1].imshow(img_grad)
        ax[1].set_title('Perturbation', fontsize=20)
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        
        ax[2].imshow(img_adv)
        ax[2].set_title('Adv. Image', fontsize=20)
        
        ax[0].axis('off')
        ax[2].axis('off')
        
        class_names = pickle.load(open('pickled_id_label_imagenet_map', 'rb'))
        # print(class_names)
        ax[0].text(1.2,0.5, "+{}*".format(epsilon), size=18, ha="center", 
                 transform=ax[0].transAxes)
        # print(int(true_label))
        # print(class_names[8])
        true_label = class_names[int(true_label)]['label']
        pred_unpert_label = class_names[int(pred_unpert_label)]['label']
        adv_label = class_names[int(adv_label)]['label']
        unpert_pred_prob = float(unpert_pred_prob[0])
        # unpert_pred_prob = float(unpert_pred_prob)
        adv_prob = float(adv_prob[0])
        

        if topkProb is not None:
            top_labels = [class_names[int(adv_label)]['label'] for  adv_label in topkLabel]


        ax[0].text(0.5,-0.4, "True Label: {} \n Prediction: {} \n Pred_Prob: {}".format(true_label,pred_unpert_label,round(unpert_pred_prob,4)), 
                   size=10, ha="center", transform=ax[0].transAxes)
        
      
        ax[1].text(1.2,0.5, " = ", size=12, ha="center", transform=ax[1].transAxes)

        if topkProb is None:
            ax[2].text(0.5,-0.3, "Adv_Pred: {}\n Adv_Prob: {}".format(adv_label, round(adv_prob,4)), size=15, ha="center", 
                 transform=ax[2].transAxes)
        else:
            start_y = -0.3
            for idx,lab in enumerate(top_labels):
                ax[2].text(0.5,start_y-(idx*0.2), "Rank : {} Pred_class: {}\n Confidence: {} \n".format((idx+1),lab, round(topkProb[idx],4)), size=10, ha="center", 
                 transform=ax[2].transAxes)

    else:
        figure, ax = plt.subplots(1,2, figsize=(10,10))
        plt.subplots_adjust(wspace=0.4)
        
        
    #     print(img.shape)
        ax[0].imshow(img)
        ax[0].set_title('Clean Image', fontsize=20)
            
        ax[1].imshow(img_adv)
        ax[1].set_title('Adv. Image', fontsize=20)
        
        ax[0].axis('off')
        ax[1].axis('off')
        
        class_names = pickle.load(open('pickled_id_label_imagenet_map', 'rb'))
        # print(class_names)

        # ax[0].text(1.2,0.5, "+{}*".format(epsilon), size=18, ha="center", 
        #          transform=ax[0].transAxes)

        # print(int(true_label))
        # print(class_names[8])
        true_label = class_names[int(true_label)]['label']
        pred_unpert_label = class_names[int(pred_unpert_label)]['label']
        adv_label = class_names[int(adv_label)]['label']
        unpert_pred_prob = float(unpert_pred_prob[0])
        # unpert_pred_prob = float(unpert_pred_prob)
        adv_prob = float(adv_prob[0])
        

        if topkProb is not None:
            top_labels = [class_names[int(adv_label)]['label'] for  adv_label in topkLabel]


        ax[0].text(0.5,-0.4, "True Label: {} \n Prediction: {} \n Pred_Prob: {}".format(true_label,pred_unpert_label,round(unpert_pred_prob,4)), 
                   size=10, ha="center", transform=ax[0].transAxes)
        
      
        # ax[1].text(1.2,0.5, " = ", size=12, ha="center", transform=ax[1].transAxes)

        if topkProb is None:
            ax[1].text(0.5,-0.3, "Adv_Pred: {}\n Adv_Prob: {}".format(adv_label, round(adv_prob,4)), size=15, ha="center", 
                 transform=ax[1].transAxes)
        else:
            start_y = -0.3
            for idx,lab in enumerate(top_labels):
                ax[1].text(0.5,start_y-(idx*0.2), "Rank : {} Pred_class: {}\n Confidence: {} \n".format((idx+1),lab, round(topkProb[idx],4)), size=10, ha="center", 
                 transform=ax[1].transAxes)



    savefig('demo.png',bbox_inches='tight')
    plt.show()