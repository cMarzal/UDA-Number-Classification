import os
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

#MEAN=[0.485, 0.456, 0.406]
#STD=[0.229, 0.224, 0.225]

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):

    def __init__(self, args, type='mnistm', mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.dir = os.path.join('../hw3_data/digits', type)
        self.data_dir = os.path.join(self.dir, mode)

        self.img_list = os.listdir(self.data_dir)
        self.img_dir = [self.data_dir + '/' + photo for photo in self.img_list]

        df = pd.read_csv(os.path.join(self.dir, mode+'.csv'), usecols=['label'])

        self.labels = df.values

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        #get data
        img_path = self.img_dir[idx]
        #read image
        img = Image.open(img_path).convert('RGB')
        #get smiling value
        lb = self.labels[idx]

        return self.transform(img), lb