#! /usr/bin/env python

from code import simCLR

### Location of the training and test datasets
training_dataset='example_images/train.txt'
test_dataset='example_images/test.txt'
datasets=[training_dataset, test_dataset]

### You can set the base_encoder to any of the ResNet models
base_encoder='resnet18'
pretrained=True
                                 
### The larger the batch_size the better.
### However, you may also be limited by your hardware, as I was
batch_size=100

num_workers=1
epochs=100

### You can change these to fit your needs. However, you'll need at least one
### complete training run before you can set train to False. If train is False
### the most recent results from your model will be loaded in order to plot
### and/or train a linear classifier on top of the base encoder.
train=True
plot_tsne=True
linear_eval=True

### Initialize the model
model=simCLR(datasets=datasets, base_encoder=base_encoder, 
             pretrained=pretrained, batch_size=batch_size,
             num_workers=num_workers, epochs=epochs, train=train,
             plot_tsne=plot_tsne, linear_eval=linear_eval)

### You can also call with the results directory you're interested in looking
### at instead of the most recent results (i.e. model('results/model_4'))
model()
