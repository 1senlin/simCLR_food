from torchvision import transforms
from scipy.ndimage import gaussian_filter
import numpy as np

### From https://arxiv.org/pdf/2002.05709.pdf
def get_color_distortion(s=1.0):
# s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
    
def gauss_blur(image):
    sig=[.1, 2.0][np.random.randint(0,2)]
    if np.random.random()>0.5:
        image=gaussian_filter(image, sigma=(sig, sig, 0))
    return image

def totensor():
    #tens=transforms.Compose([transforms.ToTensor(),
    #                         transforms.Normalize([0.485, 0.456, 0.406], 
    #                                              [0.229, 0.224, 0.225])])  
    tens=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], 
                                                  [0.5, 0.5, 0.5])])
    return tens  

def aug_data(image):
    trans=transforms.Compose([transforms.RandomResizedCrop(224),
                              get_color_distortion(1)])
    
    #aug1=transforms.RandomResizedCrop(224)(image)
    #aug2=get_color_distortions()(aug1)
    
    aug=trans(image)
    #aug=gauss_blur(aug)
    aug_image=totensor()(aug)
    return aug_image
    