import json
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

num_filters = params["num_filters"]
dropout_rate = params["dropout_rate"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

model.load_state_dict(
    torch.load(
        "models/model.pth",
        map_location=device
    )
)

model.eval()

test_data = torch.load(
    "data/processed/test.pt"
)

x_test = test_data["x"].to(device)
y_test = test_data["y"].to(device)

criterion = nn.CrossEntropyLoss()

all_predictions = []

total_loss = 0
correct = 0

batch_size = 256

with torch.no_grad():

    for i in range(
        0,
        len(x_test),
        batch_size
    ):

        images = x_test[i:i + batch_size]
        labels = y_test[i:i + batch_size]

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        correct += (
            predictions == labels
        ).sum().item()

accuracy = correct / len(y_test)

test_loss = total_loss / len(y_test)

metrics = {
    "test_accuracy": accuracy,
    "test_loss": test_loss
}

with open(
    "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

cm = confusion_matrix(
    y_test.cpu().numpy(),
    all_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()

print(metrics)