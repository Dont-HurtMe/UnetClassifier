from Unet import UnetClassifier
from Unet import train_step
from Unet import validation_step
from Unet import save_model
import torch.optim as optim
import torch
from tqdm import tqdm

def fit(train_loader,
        val_loader,
        loss_func,
        num_epochs,
        learning_rate=0.0001,
        model_save=False,
        update_lr=False):
    
    max_score = 0
    
    loss_train = []
    loss_val = []
    loss_acc = []        
   
    device = 'cuda'
    
    model = UnetClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        model.train()
        loop = tqdm(train_loader)
    
        # train model epoch = i
        for inputs, targets in loop:
        
            # define all data in gpu
            inputs ,targets = inputs.to(device) ,targets.to(device)
        
            # forward
            label = model(inputs)
            loss = loss_func(label, targets)

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
        
        overall_loss = running_loss / len(train_loader)
        val = validation_step(val_loader, model, loss_func , device)
        
        loss_train.append(overall_loss)
        loss_val.append(val['val_loss'])
        loss_acc.append(val['val_acc'])
        
        if model_save : 
            if  max_score < val['val_acc'] :
                max_score = val['val_acc']
                save_model(model)
        
        if update_lr : 
            if epoch == num_epochs//2 :
                optimizer.param_groups[0]['lr'] = 0.000001
                print('Decrease decoder learning rate to 0.0001')
                
        print(f"Epoch {epoch + 1}/{num_epochs} => "
              f"Train Loss: {overall_loss:.4f},\nValidation Loss: {val['val_loss']:.4f} , Validate Accuracy {val['val_acc']:.4f} ")
        
        print("---"*30) 
        
    return model , loss_train , loss_val , loss_acc

