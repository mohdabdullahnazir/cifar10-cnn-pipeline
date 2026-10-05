from pathlib import Path
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

train_data = CIFAR10(
    root="data/downloads",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = CIFAR10(
    root="data/downloads",
    train=False,
    download=True,
    transform=ToTensor()
)

x_train = torch.stack([image for image, _ in train_data])
y_train = torch.tensor([label for _, label in train_data])

x_test = torch.stack([image for image, _ in test_data])
y_test = torch.tensor([label for _, label in test_data])

torch.save(
    {"x": x_train, "y": y_train},
    RAW_DIR / "train.pt"
)

torch.save(
    {"x": x_test, "y": y_test},
    RAW_DIR / "test.pt"
)

print("Raw CIFAR-10 dataset saved successfully.")