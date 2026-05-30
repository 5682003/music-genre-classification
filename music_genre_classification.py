"""
Task 3 — Music Genre Classification
Model: CNN + RNN (LSTM) Hybrid
Dataset: GTZAN (10 genres)
Audio → Mel Spectrogram → CNN → LSTM → Output
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request
import tarfile
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SAMPLE_RATE  = 22050
DURATION     = 30        # seconds per clip
N_MELS       = 128       # mel filterbanks
HOP_LENGTH   = 512
N_FFT        = 2048
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 0.001
VAL_SPLIT    = 0.2
NUM_CLASSES  = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Download GTZAN Dataset
# ─────────────────────────────────────────────
DATA_DIR = Path("./gtzan")

def download_gtzan():
    DATA_DIR.mkdir(exist_ok=True)
    genres_dir = DATA_DIR / "genres_original"
    if not genres_dir.exists():
        print("Downloading GTZAN Dataset (~1.2 GB)...")
        url   = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres_original.tar.gz"
        tfile = DATA_DIR / "gtzan.tar.gz"
        urllib.request.urlretrieve(url, tfile)
        print("Extracting...")
        with tarfile.open(tfile, "r:gz") as t:
            t.extractall(DATA_DIR)
        tfile.unlink()
        print("Dataset ready!")
    else:
        print("Dataset already downloaded.")

download_gtzan()

# ─────────────────────────────────────────────
# 2. Mel Spectrogram Transform
# ─────────────────────────────────────────────
mel_transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ),
    T.AmplitudeToDB(top_db=80),
)

# ─────────────────────────────────────────────
# 3. Custom Dataset
# ─────────────────────────────────────────────
class GTZANDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []
        self.genre_to_idx = {g: i for i, g in enumerate(GENRES)}

        genres_dir = self.root_dir / "genres_original"
        for genre in GENRES:
            genre_dir = genres_dir / genre
            for audio_file in genre_dir.glob("*.wav"):
                self.samples.append((audio_file, self.genre_to_idx[genre]))

        print(f"Found {len(self.samples)} audio files across {NUM_CLASSES} genres.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != SAMPLE_RATE:
                waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad or trim to fixed length
            target_len = SAMPLE_RATE * DURATION
            if waveform.shape[1] < target_len:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_len]

            # Mel Spectrogram
            mel = mel_transform(waveform)  # (1, N_MELS, Time)
            mel = mel.squeeze(0)           # (N_MELS, Time)

            return mel, label

        except Exception:
            # Return zeros if file is corrupted
            mel = torch.zeros(N_MELS, SAMPLE_RATE * DURATION // HOP_LENGTH)
            return mel, label


dataset = GTZANDataset(DATA_DIR)

val_size   = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training  : {train_size} | Validation: {val_size}")

# ─────────────────────────────────────────────
# 4. CNN + RNN Hybrid Model
# ─────────────────────────────────────────────
class MusicGenreCNN_RNN(nn.Module):
    """
    CNN extracts local features from spectrogram frames.
    RNN (LSTM) models temporal patterns across time.
    """
    def __init__(self, num_classes=10):
        super(MusicGenreCNN_RNN, self).__init__()

        # ── CNN Feature Extractor ─────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # H/2, W/2
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # H/4, W/4
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),       # H/8, W/8
            nn.Dropout2d(0.25),
        )

        # ── RNN Temporal Modelling ────────────
        # After CNN: freq dim = N_MELS // 8 = 16
        cnn_out_freq = N_MELS // 8
        self.rnn_input_size = 128 * cnn_out_freq   # channels × freq

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # ── Classifier ───────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),   # ×2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, N_MELS, Time)
        x = x.unsqueeze(1)             # (B, 1, N_MELS, Time)

        x = self.cnn(x)                # (B, 128, N_MELS//8, Time//8)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)     # (B, T, C, F)
        x = x.reshape(B, T, C * F)    # (B, T, C*F) → RNN input

        x, _ = self.rnn(x)            # (B, T, 512)
        x = x[:, -1, :]               # last timestep → (B, 512)

        x = self.classifier(x)        # (B, num_classes)
        return x


model = MusicGenreCNN_RNN(NUM_CLASSES).to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total:,}")

# ─────────────────────────────────────────────
# 5. Loss, Optimizer & Scheduler
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─────────────────────────────────────────────
# 6. Training & Evaluation
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(labels).sum().item()
        total         += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    return running_loss / total, 100.0 * correct / total


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

print("\n── Training ─────────────────────────────────────────────")
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
print("─" * 55)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>9.4f} {val_acc:>7.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_music_genre_model.pth")

print(f"\n✓ Best validation accuracy: {best_val_acc:.2f}%")
print("✓ Model saved to best_music_genre_model.pth")

# ─────────────────────────────────────────────
# 7. Per-Genre Accuracy
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("best_music_genre_model.pth", map_location=DEVICE))
model.eval()

genre_correct = [0] * NUM_CLASSES
genre_total   = [0] * NUM_CLASSES

with torch.no_grad():
    for specs, labels in val_loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)
        outputs = model(specs)
        _, predicted = outputs.max(1)
        for label, pred in zip(labels, predicted):
            genre_correct[label] += (label == pred).item()
            genre_total[label]   += 1

print("\nPer-Genre Accuracy:")
for i, genre in enumerate(GENRES):
    if genre_total[i] > 0:
        acc = 100.0 * genre_correct[i] / genre_total[i]
        print(f"  {genre:<12}: {acc:.1f}%")

# ─────────────────────────────────────────────
# 8. Training Curves
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Validation")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"],   label="Validation")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("Music Genre Classification — CNN + RNN Hybrid", fontsize=12)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved!")

# ─────────────────────────────────────────────
# 9. Visualise Sample Spectrograms
# ─────────────────────────────────────────────
dataiter     = iter(val_loader)
specs, labels = next(dataiter)

with torch.no_grad():
    outputs  = model(specs.to(DEVICE))
    _, preds = outputs.max(1)

fig, axes = plt.subplots(2, 5, figsize=(18, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(specs[i].numpy(), aspect="auto", origin="lower", cmap="magma")
    true_genre = GENRES[labels[i]]
    pred_genre = GENRES[preds[i].cpu()]
    color      = "green" if true_genre == pred_genre else "red"
    ax.set_title(f"T: {true_genre}\nP: {pred_genre}", fontsize=8, color=color)
    ax.axis("off")

plt.suptitle("Mel Spectrograms — Genre Predictions (green=correct, red=wrong)", fontsize=11)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("Sample predictions saved!")
