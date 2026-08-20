import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_loader import GalaxyDataset, get_transforms
import os
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder():
    dataset = GalaxyDataset('/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter/Data/metadata.csv', '/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter/Data/images_gz2/images', transform=get_transforms())
    # get indices where label is 0 or 1 (normal)
    normal_indices = [i for i, (_, label) in enumerate(dataset) if label in (0,1)]
    normal_subset = Subset(dataset, normal_indices)
    loader = DataLoader(normal_subset, batch_size=32, shuffle=True)

    model = Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for images, _ in tqdm(loader, desc=f'AE Epoch {epoch+1}'):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"AE loss: {total_loss/len(loader):.4f}")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), '/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter/Models/autoencoder.pth')
    print("Autoencoder saved to /Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter/Models/autoencoder.pth")

if __name__ == '__main__':
    train_autoencoder()











