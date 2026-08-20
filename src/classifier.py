import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

from data_loader import GalaxyDataset, get_transforms


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 3

PROJECT_ROOT = "/Users/ashwi/Downloads/MQAIS/Events/Cosmic hunter"

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "metadata.csv"
)

IMAGE_ROOT = os.path.join(
    PROJECT_ROOT,
    "Data",
    "images_gz2",
    "images"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "classifier_resnet50.pth"
)


# ============================================================
# TRAINING
# ============================================================

def train():

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    print("\nChecking paths...")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    if not os.path.exists(IMAGE_ROOT):
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_ROOT}"
        )

    print("Metadata:", METADATA_PATH)
    print("Images:", IMAGE_ROOT)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = GalaxyDataset(
        METADATA_PATH,
        IMAGE_ROOT,
        transform=get_transforms()
    )

    print("Total images:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no images.")

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print("Training images:", len(train_ds))
    print("Validation images:", len(val_ds))

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load pretrained ResNet50
    # --------------------------------------------------------

    print("\nLoading pretrained ResNet50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Replace final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    print("Model loaded.")
    print("Classes:", NUM_CLASSES)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"{'=' * 60}"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        progress = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1}"
        )

        for images, labels in progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            _, predictions = torch.max(
                outputs,
                1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        train_accuracy = (
            100.0 *
            train_correct /
            train_total
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                _, predictions = torch.max(
                    outputs,
                    1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= len(val_loader)

        val_accuracy = (
            100.0 *
            val_correct /
            val_total
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()