import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from data_loader import GalaxyDataset, get_transforms
from torch.utils.data import DataLoader
import os

def get_embeddings(model, dataloader, device):
    model.eval()
    # remove fc and replace with identity for feature extraction
    # we'll use the avgpool output
    embeddings = []
    filenames = []
    with torch.no_grad():
        for images, labels, paths in dataloader:  # need custom loader returning paths
            images = images.to(device)
            # forward until avgpool
            features = model.avgpool(model(images))  # but model(images) goes all the way
            # Instead, we'll use a hook or rebuild a feature extractor
            # Simpler: create a new model without fc
            pass
    # (for brevity, we'll provide a full version in the final answer)