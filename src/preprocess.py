from pathlib import Path
import torch
import yaml

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

val_size = params["preprocess"]["val_size"]
seed = params["preprocess"]["seed"]

torch.manual_seed(seed)

train_data = torch.load("data/raw/train.pt")
test_data = torch.load("data/raw/test.pt")

x = train_data["x"].float()
y = train_data["y"].long()

x_test = test_data["x"].float()
y_test = test_data["y"].long()

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

x = (x - mean) / std
x_test = (x_test - mean) / std

indices = torch.randperm(len(x))

val_count = int(len(x) * val_size)

val_indices = indices[:val_count]
train_indices = indices[val_count:]

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {"x": x[train_indices], "y": y[train_indices]},
    processed_dir / "train.pt"
)

torch.save(
    {"x": x[val_indices], "y": y[val_indices]},
    processed_dir / "val.pt"
)

torch.save(
    {"x": x_test, "y": y_test},
    processed_dir / "test.pt"
)

print("Preprocessing completed successfully.")