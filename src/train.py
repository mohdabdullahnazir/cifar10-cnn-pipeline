from pathlib import Path
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

num_filters = params["num_filters"]
dropout_rate = params["dropout_rate"]
learning_rate = params["learning_rate"]
epochs = params["epochs"]
batch_size = params["batch_size"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_data = torch.load("data/processed/train.pt")

dataset = TensorDataset(
    train_data["x"],
    train_data["y"]
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters, num_filters * 2, 3, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 2 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = CNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

Path("models").mkdir(exist_ok=True)

history = []

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.4f}"
    )

    history.append(
        [epoch + 1, average_loss, accuracy]
    )

torch.save(
    model.state_dict(),
    "models/model.pth"
)

with open(
    "models/history.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["epoch", "loss", "accuracy"]
    )

    writer.writerows(history)

print("Training completed successfully.")