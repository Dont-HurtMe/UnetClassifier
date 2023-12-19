import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

class CustomGenerator(torch.utils.data.Dataset):
    
    def __init__(self, dataframe, transform=None):
        
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        
        return len(self.dataframe)

    def __getitem__(self, index):
        
        annotation = self.dataframe.iloc[index, 0]
        image = Image.open(annotation).convert('RGB')
        label = label = torch.tensor(int(self.dataframe.iloc[index, 3]))
        
        if self.transform:
            image = self.transform(image)
            
        return (image,label)
    
def LoadData(custom_data,
             batch_size,
             shuffle=True):
    
    return DataLoader(custom_data, 
                      batch_size=batch_size, 
                      shuffle=shuffle)

def transformer():
    return transforms.Compose([transforms.Resize((256, 256)),
                               transforms.ToTensor(),])

def get_data(dataframe,
             batch_size=32,
             shuffle=True):
    
    return LoadData(CustomGenerator(dataframe, transform=transformer()),batch_size,shuffle)
    
    
    