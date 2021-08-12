
import os
import random
import torch
from PIL import Image

class ChestXRayDataset(torch.utils.data.Dataset):
        # image_dirs is a dictionary for 3 classes (covid, normal, viral)
        # transform object for data augmentation
        def __init__(self,image_dirs, transform):
            def get_images(class_name):
                # class name is either covid, normal or viral
                
                # get the images from image_dirs using class_name
                images = [image for image in os.listdir(image_dirs[class_name])]
                print("Total number of {} images: {}".format(class_name,len(images)))
                
                return images
            # dictionary to keep track of all the images
            self.images = {}
            
            # class names
            self.class_names = ['covid','normal','viral']
            
            for name in self.class_names:
                self.images[name] = get_images(name)
            
            # save the dirs and transform
            self.image_dirs = image_dirs
            self.transform = transform
        
        # returns the length of the dataset
        # i.e. number of images in all the three classes
        def __len__(self):
            return sum([ len(self.images[name]) for name in self.class_names])
        
        # given an index return the corresponding transformed image
        def __getitem__(self,index):
            
            # randomly choose any one of the 3 classes
            class_name = random.choice(self.class_names)
            
            # ensure that the index is in the bound of number of images in that class
            # used modulo reduction
            index = index % len(self.images[class_name])
            
            # get the image name
            image_name = self.images[class_name][index]
            
            # get the image path
            image_path = os.path.join(self.image_dirs[class_name],image_name)
            
            # Load the image, convert it to RGB
            image = Image.open(image_path).convert('RGB')
            
            # return transformed version of the image and the label (0,1,2)
            return self.transform(image), self.class_names.index(class_name)
            
