from code import aug_data, FoodDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

a=FoodDataset('image_files/mini_train.txt', aug=True)
im1, im2= a[np.random.randint(0,len(a))]

a.display_image(image=(im1.numpy()*255).astype(np.uint8).transpose((1,2,0)))
a.display_image(image=(im2.numpy()*255).astype(np.uint8).transpose((1,2,0)), show=True)

if False:
    a=FoodDataset('image_files/mini_train.txt', aug=False)
    im, _= a[0]
    
    im1=np.array(im)
    im2=im1/255.0
    
    a.display_image(image=im2, show=True)