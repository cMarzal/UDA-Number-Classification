import os
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

#Set mean and STD  for normalization
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):

    def __init__(self, folder_dir):

        #set up basic parameters for dataset
        self.data_dir = os.path.join(folder_dir)
        self.img_list = sorted(os.listdir(self.data_dir))
        self.img_dir = [self.data_dir + '/' + photo for photo in self.img_list]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        #get data
        img_path = self.img_dir[idx]
        #read image
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), os.path.basename(img_path)