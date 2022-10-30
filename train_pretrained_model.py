from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from pathlib import Path
from typing import List
from PIL import Image 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


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


def main():
    """
    """
    # Hyper parameters
    batch_size = 32
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
        transforms.RandomResizedCrop(h),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_dataset = CIFAR10Dataset(
        data_path='data/train', 
        transform=train_tranforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test dataset
    test_tranforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = CIFAR10Dataset(
        data_path='data/test', 
        transform=test_tranforms
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model 
    # is_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vgg = models.vgg16(pretrained=True)
    fc_features = model_vgg.fc.in_features
    model_vgg.fc = nn.Linear(fc_features, 10)
    model_vgg = model_vgg.to(device)

    print(model_vgg.named_parameters)
    print(summary(model_vgg, input_size=(h*w*c, )))
    # if is_gpu:
    #     model_vgg.cuda()
    # computes softmax and then the cross entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_vgg.parameters(), lr=learning_rate, momentum=0.9)

    # Training 
    for epoch in range(epochs):
        train_sum_loss = 0 
        train_sum_acc = 0
        test_sum_loss = 0 
        test_sum_acc = 0
        model_vgg.train()
        for x, y in train_loader:
            x.to(device)
            y.to(device)
            # Compute output
            logit = model_vgg(x)
            loss = criterion(logit, y)
            train_sum_loss += loss.item()
            _, pred = torch.max(logit, 1)
            train_sum_acc += (pred==y).float().mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%1 == 0:
            model_vgg.eval()
            for x_test, y_test in test_loader:
                x_test.to(device)
                y_test.to(device)
                with torch.no_grad():
                    logit = model_vgg(x_test)
                    try:
                        loss = criterion(logit, y_test)
                    except:
                        print(logit.size, y_test.size)
                    test_sum_loss += loss.item()
                _, pred = torch.max(logit, 1)
                test_sum_acc += (pred==y_test).float().mean()
            print('Epoch {}: Train loss: {} -- Test loss: {} -- Train Acc: {} -- Test Acc: {}'.format(
                epoch, train_sum_loss/len(train_loader), test_sum_loss/len(test_loader),
                train_sum_acc/len(train_loader), test_sum_acc/len(test_loader)
            ))

    # Saving model 
    torch.save(model_vgg.state_dict(), result_dir)


if __name__ == '__main__':
    """
    """
    main()