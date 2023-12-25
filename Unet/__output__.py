import torch
import torch.nn as nn

from .__tools__ import *
from .__model__ import *

import matplotlib.pyplot as plt
import numpy as np

class get_model() : 
    
    def __init__(self,device=None):
        
        self.device = 'cuda'
        self.model = UnetClassifier().to(self.device)
        self.model.load_state_dict(torch.load('../example/Unet/save-model/full-epochs-model-unet-classifier.pth'))
        

    def Classifier(self,image):

        image = clean_image(image).to(self.device)
        seg , label = self.model.predict(image)
        
        # label
        label = (torch.max(torch.exp(label), 1)[1]).data.cpu().numpy()

        return self.Plot(seg,label,image)        
        
    def Classes(self,reverse=False):
        
        if reverse :
            return {0:'normal',1:'benign',2:'malignant'}
        else :
            return {'normal':0,'benign':1,'malignant':2}
        
    def Plot(self,
             seg_pred,
             label_pred,
             image):
        
        label_pred = self.Classes(reverse=True)[label_pred.item()]
        fig, ax = plt.subplots(figsize=(5, 5))
        
        ax.set_title(label_pred)        
        ax.imshow(Polygons(np.squeeze(seg_pred.cpu().detach().numpy()),0.0,0.01))#(Polygons(np.squeeze(seg_pred.cpu().detach().numpy()),0.0,0.01))
        ax.imshow(image[0].cpu().detach().numpy().squeeze().transpose((1, 2, 0)),alpha=0.5)    
            
