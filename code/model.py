from code import FoodDataset

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
import numpy as np
import os

class simCLR():
    def __init__(self, datasets=['example_images/train.txt', 
                                 'example_images/test.txt'],
                                 base_encoder='resnet18', pretrained=True,
                                 batch_size=100, num_workers=1, epochs=100,
                                 train=True, plot_tsne=True, linear_eval=True):
        
        ### First set the parameters
        self.base_encoder=base_encoder
        self.pretrained=pretrained
        self.num_epochs=epochs
        self.training=train
        self.plot=plot_tsne
        self.linear_eval=linear_eval
        
        ### Determine if GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")
        print(self.device)
        
        ### Load the datasets
        self.load_data(datasets, batch_size, num_workers)
        
        ### Next initialize the model
        self.init_model(base_encoder, pretrained) 

    def __call__(self, result_dir=None):
        if self.training:
            self.train_model()
        else:
            try:
                self.workdir=result_dir
                self.model.load_state_dict(torch.load(os.path.join(self.workdir,
                                                               'model.pth')))
            except Exception:
                self.get_working_dir()
                self.model.load_state_dict(torch.load(os.path.join(self.workdir,
                                                               'model.pth')))
            print('Most recent results loaded')

        if self.plot:
            self.plot_tsne()
            
        if self.linear_eval:
            self.linear()
    
    def load_data(self, datasets, batch_size=100, num_workers=1):
        ### Load the dataset(s). If only one dataset is provided set plotting
        ### and linear eval to False. One training datset and one test can be
        ### provided.
        
        transformations=transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], 
                                                         [0.5, 0.5, 0.5])])
        
        if type(datasets) is list or type(datasets) is tuple:
            self.train=FoodDataset(datasets[0], aug=True)
            self.test=FoodDataset(datasets[1], transform=transformations)
            
            self.dataloader_train=DataLoader(self.train, batch_size=batch_size,
                                             shuffle= True, 
                                             num_workers=num_workers)
            self.dataloader_test=DataLoader(self.test, batch_size=batch_size,
                                             shuffle= True, 
                                             num_workers=num_workers)
            if self.linear_eval:
                self.train_lin=FoodDataset(datasets[0], 
                                           transform=transformations)
                self.dataloader_lin=DataLoader(self.train_lin, 
                                               batch_size=batch_size, 
                                               shuffle= True,
                                               num_workers=num_workers)
            
        else:
            self.train=FoodDataset(datasets)
            self.dataloader_train=DataLoader(self.train, batch_size=batch_size,
                                             shuffle= True, 
                                             num_works=num_workers)
            
            self.linear_eval=False
            self.plot=False
    
    def init_model(self, base_encoder, pretrained):
        ### Currently available models
        models={'resnet18': torchvision.models.resnet18,
                'resnet34': torchvision.models.resnet34,
                'resnet50': torchvision.models.resnet50,
                'resnet101': torchvision.models.resnet101,
                'resnet152': torchvision.models.resnet152}
        
        try:
            self.model=models[base_encoder](pretrained=pretrained)
            
        except Exception:
            raise KeyError('Currently only ResNet is supported as a base '
                           'encoder')
            
        ### Remove the final linear layer and add in a small MLP
        mlp= nn.Sequential(nn.Linear(self.model.fc.in_features, 256),
                           nn.ReLU(),
                           nn.Linear(256, 128),
                           nn.ReLU(),
                           nn.Linear(128, 64))
        self.model.fc=mlp
        self.model.to(self.device)
        
        ### Init the optimizer
        self.optimizer=optim.SGD(self.model.parameters(), 
                                 lr=0.001, momentum=0.9)
        
        ### If pretrained, set a scheduler
        if pretrained:
            from torch.optim import lr_scheduler
            self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                                 step_size=10, gamma=0.05)
        else:
            self.scheduler = None
            
        
    def train_model(self):
        ### Get the directory for saving results
        self.get_next_working_dir()
        
        self.model.train()
        loss_avg=[]
        for epoch in range(self.num_epochs):
            
            print('Epoch: %g of %g'%(epoch, self.num_epochs))
            running_loss=[]
                      
            for _, images in enumerate(self.dataloader_train):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                xi=images['xi']
                xi=xi.to(self.device)
                
                xj=images['xj']
                xj=xj.to(self.device)
                
                zi=self.model(xi)
                zj=self.model(xj)
                
                loss=self.loss_func(zi, zj)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss.append(loss.item())
            
            ### Calculate average loss over the course of the run
            loss_avg.append(np.array(running_loss).mean())
            print('running loss: %g'%loss_avg[-1])
            
            if self.scheduler:
                self.scheduler.step()
            
            ### Plot the losses as we go (I usually cancel runs so this means
            ### I don't have to wait for the run to finish before saving the 
            ### progressing)
            fig = plt.figure(figsize=(6, 6))
            plt.plot(loss_avg)
            plt.title('Training Losses')
            plt.savefig(os.path.join(self.workdir, 'losses.jpg'))
            plt.close()
            
            ### Save the model each epoch for the same reasons as above
            torch.save(self.model.state_dict(), 
                       os.path.join(self.workdir, 'model.pth'))
            torch.save(self.optimizer.state_dict(), 
                       os.path.join(self.workdir, 'optimizer.pth'))
            ### Save the losses just in case
            np.savez(os.path.join(self.workdir, 'losses'), 
                     np.array(loss_avg))

    def plot_tsne(self, data=False):
        ### Make sure to throw out the MLP before plotting
        try:
            self.model.fc=self.model.fc[0]
        except Exception:
            pass
        
        self.model.eval()
        
        output_for_tsne=torch.empty((0,self.model.fc.out_features))
        labels_for_tsne=torch.empty(0, dtype=int)
        for images, labels in self.dataloader_test:
            images=images.to(self.device)
            output=self.model(images)
            output_for_tsne=torch.cat([output_for_tsne, output.cpu().data])
            labels_for_tsne=torch.cat([labels_for_tsne, labels.data])
        
        y_labels=self.test.category_names[labels_for_tsne]
        ### Change variable name to adapt easier from my previous code
        y=output_for_tsne
        
        ### Import a couple more modules
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        ### Reduce dimensions of output features before
        ### t-SNE visualization, (recommendation from scikit)
        if y.shape[-1]>50:
            pca=PCA(n_components=50)
            y=pca.fit_transform(y)
            print(y.shape)
            
        if data:
            return y, y_labels
        
        tsne=TSNE(perplexity=100, n_iter=5000)
        trans=tsne.fit_transform(y)
        sns.scatterplot(trans[:,0],trans[:,1], hue=y_labels)
        plt.legend([])
        #plt.legend(list(set(y_labels)))
        plt.savefig(os.path.join(self.workdir,'tsne.jpg'))
        plt.close()
        
    
    def linear(self, num_epochs=28):
        ### Make sure to throw out the MLP before training linear layer
        try:
            self.model.fc=self.model.fc[0]
        except Exception:
            pass
        
        ### Make sure we're only training the linear layer
        self.model.eval()
        
        ### Set up the linear layer
        output_features=self.model.fc.out_features
        self.lin_layer=nn.Linear(output_features, 
                                 len(self.test.category_names))
        self.lin_layer.to(self.device)
        
        ### Set up optimizer and loss function
        optimizer = optim.SGD(self.lin_layer.parameters(), 
                              lr=0.001, momentum=0.9)
        criterion=nn.CrossEntropyLoss()
        
        ### Also set up a scheduler to decrease learning rates over time
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        dataloaders={'train': self.dataloader_lin, 
                     'test' : self.dataloader_test}
        dataset_sizes = {'train': len(self.train_lin.file_names), 
                         'test': len(self.test.file_names)}
        
        ### Train the model, based off pytorch transfer learning tutorial
        import copy
        
        best_model_wts = copy.deepcopy(self.lin_layer.state_dict())
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print('Epoch: %g of %g'%(epoch, num_epochs))
            
            for phase in ["train", "test"]:
                if phase == "train":
                    self.lin_layer.train()
                else:
                    self.lin_layer.eval()
                
                running_loss=[]
                running_correct = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device, dtype=torch.float)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    
                    ### Don't track grads if we're evaluating the linear layer
                    with torch.set_grad_enabled(phase == "train"):
                        
                        ### Run through the base encoder first
                        intermediate=self.model(inputs)
                        
                        outputs = self.lin_layer(intermediate)
                        blah, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        ### So we don't step during the evaluation phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            
                    
                    ### Keep track of losses and accuracy
                    running_loss.append(loss.item())
                    running_correct += torch.sum(preds == labels.data)
                
                epoch_loss = np.array(running_loss).mean()
                epoch_acc = running_correct.double() / dataset_sizes[phase]
                
                print('%s Loss: %.4f Acc: %.4f'%(phase.title(), 
                                                 epoch_loss, epoch_acc))
                
                ### Make sure to copy the best model
                if phase == "test" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.lin_layer.state_dict())
                    
            scheduler.step()
                    
        print('Highest Accuracy: %.2f%%'%(best_acc*100))

        ### Save the best linear model
        self.lin_layer.load_state_dict(best_model_wts)
        torch.save(self.lin_layer.state_dict(), 
                    os.path.join(self.workdir, 'linear_model.pth'))
        f=open(os.path.join(self.workdir, 'info.txt'),'a')
        f.write('Accuracy='+str(best_acc*100))
        f.close()

    def test_linear(self):
        try:
            self.model.fc=self.model.fc[0]
        except Exception:
            pass
        
        self.model.eval()
        
        try:
            self.lin_layer.load_state_dict(torch.load(os.path.join
                                                      (self.workdir, 
                                                       'linear_model.pth')))
        except Exception:
            output_features=self.model.fc.out_features
            self.lin_layer=nn.Linear(output_features, 
                                     len(self.test.category_names))
            self.lin_layer.load_state_dict(torch.load(os.path.join
                                                      (self.workdir, 
                                                       'linear_model.pth')))
        
        self.lin_layer.to(self.device)
        self.lin_layer.eval()
        
        running_correct = 0
        for inputs, labels in self.dataloader_test:
            inputs = inputs.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)
            
            intermediate=self.model(inputs)
            outputs = self.lin_layer(intermediate)
            _, preds = torch.max(outputs, 1)
            
            running_correct += torch.sum(preds == labels.data)
            
        acc=(running_correct.double()/len(self.test.file_names))*100
        print('Accuracy: %.2f%%' %acc)
        f=open(os.path.join(self.workdir, 'info.txt'),'a')
        f.write('Accuracy='+str(acc.item())+'%')
        f.close()
    
    def loss_func(self, zi, zj, temp=0.1):
        batch_size=len(zi)
    
        ### Loss function was written with guidance from..
        ### https://github.com/thunderInfy/simclr/blob/master/resnet-simclr.py
        
        ### Normalize the inputs
        zi=torch.div(zi, torch.norm(zi, dim=1).reshape(-1,1))
        zj=torch.div(zj, torch.norm(zj, dim=1).reshape(-1,1))
        z=torch.cat([zi, zj])
        
        ### Calculate the cosine similarities for the numerator
        top_ij=torch.exp(torch.div(torch.nn.CosineSimilarity()(zi, zj),temp))
        top_ji=torch.exp(torch.div(torch.nn.CosineSimilarity()(zj, zi),temp))
        top=torch.cat([top_ij,top_ji])
        
        ### Calculate all the cosine similarities for the denominator
        ### (Note z normalized, so we can simply perform matrix multiplication)
        bottom_all=torch.exp(torch.div(torch.mm(z,torch.t(z)), temp))
        ### Sum each cosine similarities, while taking into account 
        ### that i,i will equal 0 within the sum
        diagonal=torch.diag(torch.diagonal(bottom_all))
        bottom=torch.sum(bottom_all-diagonal,dim=1)
        
        ### Calculate the total loss
        l=-torch.log(torch.div(top, bottom))
        L=torch.div(torch.sum(l), 2*batch_size)
        
        return L
    
    def get_next_working_dir(self):
        ### Check if results directory has been made
        if not os.path.isdir('results'):
            os.mkdir('results')
            
        current_dirs=glob('results/model_*')
        
        if current_dirs:
            ### If there are already models saved, find the next working dir
            nums=[]
            for dir in current_dirs:
                nums.append(int(dir.split('_')[-1]))
                
            self.workdir='results/model_%g'%(max(nums)+1)
            os.mkdir(self.workdir)
        else:
            self.workdir='results/model_1'
            os.mkdir(self.workdir)
            
        with open(os.path.join(self.workdir, 'info.txt'),'w') as f:
            f.write(self.base_encoder+'\n')
            f.write('Pretrained='+str(self.pretrained))
            
    def get_working_dir(self):
        if not os.path.isdir('results'):
            raise OSError('No results files, must train model first')
        
        current_dirs=glob('results/model_*')
        
        if current_dirs:
            nums=[]
            for dir in current_dirs:
                nums.append(int(dir.split('_')[-1]))
                
            self.workdir='results/model_%g'%max(nums)
        
        
            
        
            