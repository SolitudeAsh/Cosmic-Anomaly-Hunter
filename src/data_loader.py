import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

class GalaxyDataset(Dataset):
    def __init__(self, metadata_path, img_dir, transform=None):
        self.df = pd.read_csv(metadata_path)
        self.img_dir = img_dir
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    # quick test
    ds = GalaxyDataset('/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter/Data/metadata.csv', '/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter/Data/images_gz2/images', transform=get_transforms())
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))
    print(images.shape, labels)