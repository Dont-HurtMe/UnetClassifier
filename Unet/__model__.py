import torch
import torch.nn as nn

class UnetClassifier(nn.Module):
    
    def __init__(self, num_classes=3):
        
        super(UnetClassifier, self).__init__()

        # Fully layer
        self.mask_model = torch.load('../example/Unet/unet.pth')
        
        # Freeze the parameters of mask_model
        for param in self.mask_model.parameters():
            param.requires_grad = False
            
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2**16, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.mask_model(x)
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
    
    def predict(self,x) :
           
        x = x.to('cuda')
        seg = self.mask_model(x) 
        label = self.forward(x)
    
        return seg,label 
