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

train=FoodDataset('image_files/kitty_train.txt', transform=transformations)

test=FoodDataset('image_files/kitty_test.txt', transform=transformations)

dataloader_train=torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                                        shuffle=True,
                                                        num_workers=1)
dataloader_test=torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=1)

dataloaders={'train': dataloader_train, 'test': dataloader_test}

dataset_sizes = {'train': len(train.file_names), 
                 'test': len(test.file_names)}

print(dataset_sizes)
#class_names = image_datasets["train"].classes
#print(class_names)

model = torchvision.models.resnet18(pretrained=False)

### So we don't train the pretrained model
#
#for param in model.parameters():
#    param.requires_grad = False

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

model.load_state_dict(torch.load("results/model.pth"))

model.eval()

output_features=model.fc[-1].out_features

lin_layer=nn.Linear(output_features, len(test.category_names))
lin_layer.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(lin_layer, model, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(lin_layer.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                lin_layer.train()  # Set model to training mode
            else:
                lin_layer.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    
                    intermediate=model(inputs)
                    
                    outputs = lin_layer(intermediate)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(lin_layer.state_dict())
            
        print()
    
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    lin_layer.load_state_dict(best_model_wts)
    return lin_layer

criterion = nn.CrossEntropyLoss()
lin_layer=train_model(lin_layer, model, criterion, optimizer, 100)