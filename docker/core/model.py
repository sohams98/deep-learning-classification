from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob 
import numpy as np

class ConvolutionModel(nn.Module):
    def __init__(self, input_width=640, input_height=480, num_classes=10, is_bw=False):
        super().__init__()
        channels = 1 if is_bw else 3
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(32*int((input_width/2)*(input_height/2)), 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(512, 128)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(128, 64)
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(64, num_classes)
        self.act6 = nn.LogSoftmax(dim=1)
 
    def forward(self, x):
        # input 3x640x480, output 32x640x480
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        
        # input 32x640x480, output 32x640x480
        x = self.act2(self.conv2(x))
        
        # input 32x640x480, output 3x640x480
        x = self.pool2(x)
        
        # input 32x320x240, output 2457600
        x = self.flat(x)
        
        # input 2457600, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)

        # input 512, output 128
        x = self.act4(self.fc4(x))
        # x = self.drop4(x)

        # input 128, output 10
        x = self.act5(self.fc5(x))
        x = self.drop5(x)

        x = self.act6(self.fc6(x))
        return x

class ImageDataset(Dataset):
    def __init__(self, dir, width, height, is_bw=False):
        self.dir = dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ]) if is_bw else transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ])
        self.target_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.imageNames = glob.glob(f'{dir}/*/images/*.jpg')
        self.labelNames = glob.glob(f'{dir}/*/labels/*.txt')
    
    def __len__(self):
        return len(self.imageNames)
    
    def __getitem__(self, index):
        image = Image.open(self.imageNames[index])
        with open(self.labelNames[index], 'r') as file:
            label = file.read()
            label = np.array(label)
        
        return self.transform(image), int(label)

class ImageDatasetPadded(Dataset):
    def __init__(self, dir, width, height, is_bw=False):
        self.dir = dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad(30),
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ]) if is_bw else transforms.Compose([
            transforms.Pad(30),
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ])
        self.target_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.imageNames = glob.glob(f'{dir}/*/images/*.jpg')
        self.labelNames = glob.glob(f'{dir}/*/labels/*.txt')
    
    def __len__(self):
        return len(self.imageNames)
    
    def __getitem__(self, index):
        image = Image.open(self.imageNames[index])
        with open(self.labelNames[index], 'r') as file:
            label = file.read()
            label = np.array(label)
        
        return self.transform(image), int(label)
    