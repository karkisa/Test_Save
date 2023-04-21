import torchvision 
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
import pdb
# build base pipe
# build model 
# buil logic using pl
# log the training in weightsa and biases
# train the trainer
# analyse the model performance
# improve model


class base_pipe(Dataset):

    def __init__(self, list_imgs,size=(224,224)) :
        self.list_imgs = list_imgs
        self.resize = torchvision.transforms.Resize(size=size)
        self._label_dict = {
                                "0" : 0,
                                "2" : 1,
                                "4" : 2, 
                                "6" : 3,
                                "9" :  4
                            }
                        
    def read_img_class(self, img_path):
        image = Image.open(img_path)
        image = ToTensor()(image)
        image = self.resize(image)
        cls = img_path.split('/')[-2]
        return image,self._label_dict[cls]

    def __len__(self):
        return len(self.list_imgs)
    
    def __getitem__(self, index) :

        img,cls = self.read_img_class(self.list_imgs[index])
        # add augmentation here
        # or in logic file
        return img,cls
    


