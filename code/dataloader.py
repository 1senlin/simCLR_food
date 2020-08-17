"""
Utilities to load in the image data.
"""
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from more_itertools import unique_everseen
import h5py
import os
import piexif
from code import aug_data

class FoodDataset(Dataset):
    """An Object to load in the image data

    Will load either an HDF5 file containing a stack of images
    and the corresponding food categories, or will load images
    in the specified directory. Can be indexed to return the image
    and food name

    Attributes:
        images (`numpy.ndarray`): An n x W x H x C array where n
            is the number of images, W and H are the width and height
            of the images, and C is the number of color channels.
        category (`numpy.ndarray`): An n x F boolean array where n is
            the number of images, and F is the number of food
            classifications.
        category_names (`numpy.ndarray`): A size F array containing the
            names of the food classifications.

    Args:
        filename (`str`): The file location of an HDF5 file
        transform (`torchvision.transforms.Compose`): Contains the
            transformations applied to each image prior to being imported
            to pytorch.

    """

    def __init__(self, filename, aug=False, transform=None):
        # For now it just reads the h5 file
        # Will eventually add path functionality
        if filename.endswith('h5'):
            f = h5py.File(filename, "r")
            self.images = f["images"][()]
            self.category = f["category"][()]
            self.category_names = f["category_names"][()]
            f.close()
        else:
            self.root_path = '/'.join(filename.split('/')[:-1])+'/images'
            self.file_names = pd.read_csv(filename, header=None, names=['ims'])
            self.file_names = self.file_names.apply(lambda x: x+'.jpg' if not
                                                    x.ims.endswith('.jpg') else 
                                                    x, axis=1)
            self.category_names = np.array(list(unique_everseen(self.file_names.
                                    apply(lambda x: x.ims.split('/')[0],
                                          axis=1).tolist())))
            
            self.file_names=self.file_names.ims.values
            for file in self.file_names:
                piexif.remove(os.path.join(self.root_path,file))
        
        
        self.transform=transform
        self.aug=aug

    def __getitem__(self, idx):
        try:
            image = self.images[idx, :, :, :]
            food_idx=np.argwhere(self.category[idx])[0,0]
        except Exception:
            image_name=os.path.join(self.root_path, self.file_names[idx])
            image=Image.open(image_name)
            food_idx=np.argwhere(self.category_names==
                                 image_name.split('/')[-2])[0,0]
        
        if self.aug:
            im1=aug_data(image)
            im2=aug_data(image)
            return {'xi':im1, 'xj': im2}
            
        if self.transform:
            image=self.transform(image)
        
        return image, food_idx

    def __len__(self):
        try:
            return len(self.images)
        except Exception:
            return len(self.file_names)
        
    def display_image(
        self,
        idx: Optional[int] = None,
        image: Optional[np.array] = None,
        food: Optional[str] = None,
        show: bool = False,
    ):
        """ Displays an image from the dataset

        Function will display either and image from the loaded dataset.
        If there is no user input, a random image will be displayed.

        Args:
            idx (int, optional): Index of the picture number to display
            image (numpy.ndarray, optional): Numpy array of the image to
                be displayed.
            food (str, optional): Name of the food classification.

        """

        if idx is not None:
            image, food_idx = self[idx]
            food=self.category_names[food_idx]
        elif image is None:
            idx = np.random.randint(len(self))
            image, food_idx = self[idx]
            food=self.category_names[food_idx]

        plt.figure(figsize=[4, 4])
        plt.imshow(image)

        try:
            plt.title(food.astype(str))
        except Exception:
            pass
        if show:
            plt.show()
        else:
            return plt.gcf(), plt.gca()
