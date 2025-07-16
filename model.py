import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

DATA_DIR = "gesture_data"
GESTURE_LIST = [fname.split(".")[0] for fname in os.listdir(
    DATA_DIR) if fname.endswith(".csv")]


class GestureDataset(Dataset):
    def __init__(self, data_dir, gesture_list):
        self.samples = []
        self.labels = []
        self.label_map = {gesture: idx for idx,
                          gesture in enumerate(gesture_list)}
        for gesture in gesture_list:
            filepath = os.path.join(data_dir, f"{gesture}.csv")
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 63:
                        continue
                    keypoints = np.array(row, dtype=np.float32)
                    self.samples.append(keypoints)
                    self.labels.append(self.label_map[gesture])

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

        print(
            f"Loaded {len(self.samples)} samples for {len(gesture_list)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class GestureClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=len(GESTURE_LIST)):
        super(GestureClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train():
    dataset = GestureDataset(DATA_DIR, GESTURE_LIST)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GestureClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "gesture_classifier.pth")
    print("Model saved as gesture_classifier.pth")


if __name__ == "__main__":
    train()
