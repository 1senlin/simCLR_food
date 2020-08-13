from code import FoodDataset, loss_func, plot_tsne
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sns

#####Set some parameters###
num_epochs=500
train=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_size=100

transformations=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train=FoodDataset('image_files/kitty_train.txt', aug=True)

test=FoodDataset('image_files/kitty_test.txt', transform=transformations)

dataloader_train=torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                                        shuffle=True,
                                                        num_workers=1)
dataloader_test=torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=1)

dataset_sizes = {'train': len(train.file_names), 
                 'test': len(test.file_names)}

print(dataset_sizes)
#class_names = image_datasets["train"].classes
#print(class_names)

model = torchvision.models.resnet18(pretrained=True)

in_feat = model.fc.in_features

final_mlp=nn.Sequential(nn.Linear(in_feat, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64))

## Add final layer to the model
model.fc = final_mlp
model.to(device)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, optimizer, scheduler, num_epochs):
    loss_avg=[]
    for epoch in range(num_epochs):
        
        print('epoch: %g'%epoch)
        
        model.train()
        running_loss=[]
                  
        for _, images in enumerate(dataloader_train):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            xi=images['xi']
            xi=xi.to(device)
            
            xj=images['xj']
            xj=xj.to(device)
            
            zi=model(xi)
            zj=model(xj)
            
            loss=loss_func(zi, zj)
            
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
        
        loss_avg.append(np.array(running_loss).mean())
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(loss_avg)
        plt.legend(['Training Losses'])
        plt.savefig('losses_gd.jpg')
        plt.close()
        print('running loss: %g'%loss_avg[-1])
        torch.save(model.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

if False:
    train_model(model, optimizer, scheduler, num_epochs)
else:
    model.load_state_dict(torch.load("results/model_3layer_mlp_256_64_40ish.pth"))
    optimizer.load_state_dict(torch.load("results/optimizer_3layer_mlp_256_64_40ish.pth"))

if True:
    model.eval()
    output_for_tsne=torch.empty((0,model.fc[-1].out_features))
    labels_for_tsne=torch.empty(0, dtype=int)
    for images, labels in dataloader_test:
        images=images.to(device)
        output=model(images)
        output_for_tsne=torch.cat([output_for_tsne, output.cpu().data])
        labels_for_tsne=torch.cat([labels_for_tsne, labels.data])
    
    labels=test.category_names[labels_for_tsne]
    plot_tsne(output_for_tsne, labels)
