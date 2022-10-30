from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from pathlib import Path
from typing import List
from PIL import Image 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np 

class CIFAR10Dataset(Dataset):
    """
    """
    def __init__(self, 
            data_path: str="data/train",
            transform: transforms=None
        ):
        self.classes = {i: v for i, v in enumerate(os.listdir(data_path))}
        self.label_id = {v: i for i, v in enumerate(os.listdir(data_path))}
        self.data_path = Path(data_path)
        self.transform = transform
        self._read_data()

    def _read_data(self):
        """
        """
        self.image_names = []
        self.labels = []
        for _data in os.listdir(self.data_path):
            sub_data = [f"{self.data_path}/{_data}/{file_name}" for file_name in os.listdir(self.data_path / _data)]
            self.image_names = self.image_names + sub_data
            self.labels = self.labels + [self.label_id[_data]] * len(sub_data)

    def __getitem__(self, index: int):
        image_path = self.image_names[index]
        label = self.labels[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


# Hyper parameters
batch_size = 50
epochs = 5 
learning_rate = 0.01
result_dir = 'model/vgg_model.pth'
h, w, c = 32, 32, 3
num_classes = 10

# Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
# Train dataset
train_tranforms = transforms.Compose([
    # transforms.RandomResizedCrop(h),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
])
train_dataset = CIFAR10Dataset(
    data_path='data/train', 
    transform=train_tranforms
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# # obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
print(images.shape, labels)

import matplotlib.pyplot as plt

# helper function to un-normalize and display an image
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    print('test', np.transpose(img, (1, 2, 0)).shape)
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 8))
# display 20 images
for idx in np.arange(50):
    ax = fig.add_subplot(5, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(train_dataset.classes[int(labels[idx].numpy())])
plt.show()