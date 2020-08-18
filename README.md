# simCLR_food

Simple demonstration using simCLR as a classifier for a subset of foods from the Food-101 dataset. Given software restrictions, I have simplified the problem to only 5 different foods. Even still, my trusty GTX 1060 GPU could only handle a batch size of 100 images, which is well below optimal ~4k-8k batch sizes used in the [original paper](https://arxiv.org/pdf/2002.05709.pdf).

## Use

To train your model using the dataset provided, clone the repository, install the necessary python packages, and run the provided executable. 

```bash
git clone 
cd simCLR_food
conda env create -f environment.yml
conda activate simCLR_food
./run_simCLR.py
```

If you're interested in using framework provided here to train your model using data of your own, use the *example_images* directory as well as the *train.txt* and *test.txt* files as templates for how to format your data. You should then replace the training and test dataset variables at the top of the *run_simCLR.py* with your own files in order to run the command. The results will be stored in a newly created *results* folder.

## Results

I was able to achieve ~60% accuracy by fine-tuning the base encoder (ResNet-18 in this case) and training a linear classifier on top of it. While this result is modest, the results should improve signficantly by increasing the batch-sizeto at least 248 images, and the number of training images (currently 750 total). Further, we can visualize the direct output of the base encoder using t-SNE (shown below). Here we can see decent clustering of our data, particularly the chocolate cake (yum), although the results are far from perfect. However, as this implementation of simCLR is simply for demonstration purposes, I was happy with the results seen here.

![tsne](/example_images/tsne.png)
