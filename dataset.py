import torchvision.transforms as tfs
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

def loader(path):
    img = Image.open(path)
    return img.convert('RGB')
    
def show_images(batch_images, titles=None, reverse_dict=None):
    batch_size, num_views, C, H, W = batch_images.shape

    batch_images = batch_images * STD[None, None, :, None, None] + MEAN[None, None, :, None, None]
    batch_images = batch_images.clamp(0, 1)

    fig, axes = plt.subplots(batch_size, num_views, figsize=(num_views * 2, batch_size * 2))

    for i in range(batch_size):
        for j in range(num_views):
            img = batch_images[i, j].permute(1, 2, 0).numpy() 
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            if titles is not None and j == 0:
                axes[i, j].set_title(reverse_dict[titles[i]], fontsize=12, loc='left')

    plt.tight_layout()
    plt.show()
    
class MultiViewDataset(Dataset):
    def __init__(self, dataset, num_views, image_size=256):
        self.dataset = dataset
        self.num_views = num_views
        self.transform = tfs.Compose([
            tfs.Resize((image_size, image_size)),
            tfs.RandomHorizontalFlip(),
            tfs.RandomVerticalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=MEAN, std=STD)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        views = [self.transform(img) for _ in range(self.num_views)]
        views = torch.stack(views)
        return views, label

def get_multiview_loaders(path="data/", batch_size=32, num_views=4, image_size=256, num_workers=4, pin_memory=True, split=0.8):
    base_dataset = ImageFolder(root=path, loader=loader)
    split_idx = int(len(base_dataset) * split)
    train_base, val_base = torch.utils.data.random_split(base_dataset, [split_idx, len(base_dataset) - split_idx])

    train_dataset = MultiViewDataset(train_base, num_views=num_views, image_size=image_size)
    val_dataset = MultiViewDataset(val_base, num_views=num_views, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    _, class_dict = train_base.dataset.find_classes(path)
    reverse_dict = {value: key for key, value in class_dict.items()}

    return train_loader, val_loader, class_dict, reverse_dict
