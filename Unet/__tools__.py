from tqdm import tqdm
import torch
from PIL import Image
import cv2 
import numpy as np

from .__loader__ import *


def accuracy(outputs, targets):
    """
    Compute the accuracy given predicted outputs and true targets.

    Args:
    - outputs (torch.Tensor): Predicted outputs from the model
    - targets (torch.Tensor): True labels

    Returns:
    - float: Accuracy
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def train_step(train_loader,
               model,
               loss_fn,
               optimizer,
               device):
    
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader)
    
    # train model epoch = i
    for inputs, targets in loop:
        
        # define all data in gpu
        inputs ,targets = inputs.to(device) ,targets.to(device)
        
        # forward
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient or adam step
        optimizer.step()
        
        # progress bar
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss=loss.item())
        
        # sum result
        running_loss += loss.item()
        
    return running_loss / len(train_loader)


def validation_step(valid_dl, model, loss_fn,device):
    
    model.eval()
    total_accuracy = 0.0
    total_samples = 0
    
    for image, label in valid_dl:
        
        image ,label = image.to(device) ,label.to(device)
        
        out = model(image)
        loss = loss_fn(out, label)
        acc = accuracy(out, label)

        total_samples += label.size(0)
        total_accuracy += acc * label.size(0)

    # Compute overall accuracy
    overall_accuracy = total_accuracy / total_samples
    
    return {"val_loss": loss, "val_acc": overall_accuracy} #,"val_auc": overall_auc}

def save_model(model,path='../example/Unet/save-model/best-model-unet-classifier.pth'):
    
    torch.save(model.state_dict(), path)
    print('Model saved!')
    
def clean_image(image):
    
    transform = transformer()
    resize_image = transform(image)
    resize_image = resize_image.unsqueeze(0)
    
    return resize_image.cuda()

def Polygons(img,
             epls=0,
             min_=0.1):
    _, binary_image = cv2.threshold(img, min_, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the vertices of each contour and draw polylines
    polygons = []
    
    for contour in contours:
        
        # Convert contours to polygon approximation
        epsilon = epls * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        
        polygons.append(approx_polygon)
        
    return cv2.polylines(img,polygons,isClosed=True, color=(255,0,0) ,thickness = 2)    
    
