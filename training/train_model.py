"""
PyTorch Model Training Script - GPU Accelerated
Optimized for NVIDIA RTX 4080

This script trains gesture recognition models using PyTorch with full GPU support.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class GestureDataset(Dataset):
    """
    PyTorch Dataset for gesture sequences.

    `train=True` enables on-the-fly LANDMARK-SPACE augmentation: small
    rotations / translations / scalings / per-joint noise / time-warp.
    These are cheap (no extra MediaPipe passes) and dramatically improve
    robustness of moving-sign recognition without exploding the offline
    dataset size.
    """

    def __init__(self, X, y, train: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
            x = _augment_sequence(x)
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Landmark-space augmentation
# ---------------------------------------------------------------------------

# Feature layout (must match data_preparation/feature_extractor.py).
_OFF_HAND_LOCAL = 0
_HAND_LOCAL_SIZE = 126
_OFF_FACE = 126
_FACE_SIZE = 60
_OFF_WRIST_GLOBAL = 186
_OFF_FINGERTIP_GLOBAL = 192
_OFF_WRIST_VELOCITY = 198
_OFF_WRIST_ACCEL = 204
_OFF_MASK = 210


def _augment_sequence(seq: torch.Tensor) -> torch.Tensor:
    """
    seq: (T, F). Returns augmented copy. Operates on the trajectory channels
    (which carry physical coordinates) and adds a tiny per-frame noise to
    the local-handshape channels.

    Augmentations:
        - global 2D rotation around the body-anchor origin (small angle)
        - global isotropic scaling
        - global 2D translation
        - small additive Gaussian noise on local handshape
        - time-stretch via index resampling
    """
    if seq.shape[-1] < _OFF_MASK:
        return seq  # unknown layout, skip

    out = seq.clone()
    T = out.shape[0]

    # 1) Tiny rotation about body-anchor origin (xy plane only).
    angle = (torch.rand(1).item() - 0.5) * 0.20  # +/- 0.10 rad ~ 5.7 deg
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    rot = torch.tensor([[cos, -sin], [sin, cos]], dtype=out.dtype)

    # 2) Tiny isotropic scale.
    scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.10  # +/- 5%

    # 3) Tiny 2D translation.
    tx = (torch.rand(1).item() - 0.5) * 0.04
    ty = (torch.rand(1).item() - 0.5) * 0.04

    def _xform_xy(block_off: int, n_vec: int):
        """Apply rotation+scale+translation to xy components of n_vec 3D vectors."""
        for k in range(n_vec):
            base = block_off + k * 3
            xy = out[:, base:base + 2]               # (T, 2)
            xy_new = (xy @ rot.T) * scale
            xy_new[:, 0] += tx
            xy_new[:, 1] += ty
            out[:, base:base + 2] = xy_new
            # z scales but isn't translated/rotated (monocular z is noisy).
            out[:, base + 2] *= scale

    # Trajectory channels carry real coordinates - transform them.
    _xform_xy(_OFF_WRIST_GLOBAL, 2)
    _xform_xy(_OFF_FINGERTIP_GLOBAL, 2)

    # Velocity / acceleration are differences - same rotation+scale, NO translation.
    def _xform_diff(block_off: int, n_vec: int):
        for k in range(n_vec):
            base = block_off + k * 3
            xy = out[:, base:base + 2]
            xy_new = (xy @ rot.T) * scale
            out[:, base:base + 2] = xy_new
            out[:, base + 2] *= scale

    _xform_diff(_OFF_WRIST_VELOCITY, 2)
    _xform_diff(_OFF_WRIST_ACCEL, 2)

    # 4) Local-handshape jitter (tiny - it's already wrist-centered).
    noise = torch.randn(T, _HAND_LOCAL_SIZE) * 0.01
    out[:, _OFF_HAND_LOCAL:_OFF_HAND_LOCAL + _HAND_LOCAL_SIZE] += noise

    # 5) Time-stretch via integer index resampling (probability 0.3).
    if torch.rand(1).item() < 0.3:
        warp = 0.85 + torch.rand(1).item() * 0.30  # 0.85x .. 1.15x
        new_T = max(2, int(round(T / warp)))
        idx = torch.linspace(0, T - 1, new_T).round().clamp(0, T - 1).long()
        warped = out[idx]
        if warped.shape[0] >= T:
            out = warped[:T]
        else:
            # Pad by repeating last frame.
            pad = warped[-1:].repeat(T - warped.shape[0], 1)
            out = torch.cat([warped, pad], dim=0)

    return out


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for gesture recognition."""

    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=10, dropout=0.3):
        super(LSTMModel, self).__init__()

        layers = []
        in_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.LSTM(
                in_size,
                hidden_size,
                batch_first=True,
                dropout=dropout if i < len(hidden_sizes)-1 else 0
            ))
            in_size = hidden_size

        self.lstm_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # Take last time step
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for gesture recognition (Recommended)."""

    def __init__(self, input_size, num_classes=10, dropout=0.3):
        super(CNNLSTMModel, self).__init__()

        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)

        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=dropout)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Transpose back for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM temporal modeling
        x, _ = self.lstm(x)

        # Take last time step
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNBiLSTMAttnModel(nn.Module):
    """
    CNN + Bidirectional LSTM + attention pooling.

    Recommended default for moving-sign recognition. Compared to the old
    CNN-LSTM:
      - bidirectional LSTM uses both past and future context
      - attention pooling aggregates over ALL timesteps (the old model
        only used the last hidden state, which is wasteful for sequences
        where the discriminative motion happens mid-clip)
      - extra width to handle the larger feature vector
    """

    def __init__(self, input_size, num_classes=10, dropout=0.3):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        # No max-pool: it halves the temporal resolution and we want the
        # attention pooling to see every frame.
        self.dropout_conv = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Additive attention over time.
        self.attn_proj = nn.Linear(256, 128)
        self.attn_score = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_norm(x)
        x = x.transpose(1, 2)                          # (B, F, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = x.transpose(1, 2)                          # (B, T, 192)

        x, _ = self.lstm(x)                            # (B, T, 256)

        # Attention pooling.
        attn_h = torch.tanh(self.attn_proj(x))         # (B, T, 128)
        attn_logits = self.attn_score(attn_h).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)         # (B, 256)

        pooled = self.dropout(pooled)
        return self.fc(pooled)


class TransformerModel(nn.Module):
    """Transformer model for gesture recognition."""

    def __init__(self, input_size, num_classes=10, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.3):
        super(TransformerModel, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_dataset(data_dir):
    """Load the processed dataset."""
    data_path = Path(data_dir) / 'dataset.pkl'

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Dataset loaded successfully!")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Validation samples: {len(data['X_val'])}")
    print(f"  Test samples: {len(data['X_test'])}")

    return data


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001,
                patience=15, output_dir='models', class_weights=None, grad_clip=1.0,
                weight_decay=1e-4, label_smoothing=0.1):
    """Train the model with early stopping."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_weights is not None:
        cw = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        print(f"Using class-weighted CrossEntropy. Min/max weight: {cw.min().item():.3f} / {cw.max().item():.3f}")
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # AdamW decouples weight decay from the gradient update (true L2 regularization).
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"\nStarting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, grad_clip=grad_clip,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    return history


def evaluate_model(model, test_loader, class_names, device, output_dir='models'):
    """Evaluate model on test set."""
    output_dir = Path(output_dir)

    model.eval()
    all_preds = []
    all_labels = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    test_acc = (all_preds == all_labels).mean()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")

    # Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   output_dict=True)

    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return {
        'test_accuracy': test_acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def plot_training_history(history, output_dir='models'):
    """Plot training history."""
    output_dir = Path(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history['train_acc'], label='Train')
    ax1.plot(history['val_acc'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history['train_loss'], label='Train')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300)
    plt.close()
    print(f"Training history saved to {output_dir / 'training_history.png'}")


def main():
    """
    Main function to train gesture recognition model.
    Edit config.py to change settings.
    """
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        import config
    except ImportError:
        print("Warning: config.py not found. Using default settings.\n")
        DATA_DIR = "processed_data"
        MODEL_DIR = "models"
        EPOCHS = 100
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        MODEL_TYPE = 'cnn_bilstm_attn'
        DROPOUT = 0.3
        LSTM_UNITS = [128, 64]
        USE_CLASS_WEIGHTS = True
        GRAD_CLIP_NORM = 1.0
        WEIGHT_DECAY = 1e-4
        LABEL_SMOOTHING = 0.1
    else:
        if not config.validate_config():
            print("\nPlease fix configuration errors in config.py before continuing.")
            return

        DATA_DIR = config.OUTPUT_DIR
        MODEL_DIR = config.MODEL_DIR
        EPOCHS = config.EPOCHS
        BATCH_SIZE = config.BATCH_SIZE
        LEARNING_RATE = config.LEARNING_RATE
        MODEL_TYPE = config.MODEL_TYPE
        DROPOUT = config.DROPOUT
        LSTM_UNITS = config.LSTM_UNITS
        USE_CLASS_WEIGHTS = getattr(config, 'USE_CLASS_WEIGHTS', True)
        GRAD_CLIP_NORM = getattr(config, 'GRAD_CLIP_NORM', 1.0)
        WEIGHT_DECAY = getattr(config, 'WEIGHT_DECAY', 1e-4)
        LABEL_SMOOTHING = getattr(config, 'LABEL_SMOOTHING', 0.1)

    # Load data
    data = load_dataset(DATA_DIR)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    info = data['info']
    sequence_length = info['sequence_length']
    num_features = info['landmarks_per_frame']
    num_classes = info['num_classes']
    class_names = info['classes']
    class_weights = data.get('class_weights') if USE_CLASS_WEIGHTS else None

    print(f"\nModel configuration:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features per frame: {num_features}")
    if num_features == 216:
        print(f"    - Hand local: 126 (2 x 21 x 3)")
        print(f"    - Face anchors: 60")
        print(f"    - Wrist global trajectory: 6")
        print(f"    - Fingertip global trajectory: 6")
        print(f"    - Wrist velocity: 6")
        print(f"    - Wrist acceleration: 6")
        print(f"    - Mask + reserved: 6")
    elif num_features == 186:
        print(f"    - Old layout (126 hand + 60 face). Re-run extraction for the new feature space.")
    print(f"  Number of classes: {num_classes}")
    print(f"  Model type: {MODEL_TYPE}")
    print(f"  Class weights: {'on' if class_weights is not None else 'off'}")
    print(f"  Device: {device}")

    # Create datasets - on-the-fly augmentation for training only.
    train_dataset = GestureDataset(X_train, y_train, train=True)
    val_dataset = GestureDataset(X_val, y_val, train=False)
    test_dataset = GestureDataset(X_test, y_test, train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    if MODEL_TYPE == 'lstm':
        model = LSTMModel(
            input_size=num_features,
            hidden_sizes=LSTM_UNITS,
            num_classes=num_classes,
            dropout=DROPOUT
        )
    elif MODEL_TYPE == 'cnn_lstm':
        model = CNNLSTMModel(
            input_size=num_features,
            num_classes=num_classes,
            dropout=DROPOUT
        )
    elif MODEL_TYPE == 'cnn_bilstm_attn':
        model = CNNBiLSTMAttnModel(
            input_size=num_features,
            num_classes=num_classes,
            dropout=DROPOUT,
        )
    elif MODEL_TYPE == 'transformer':
        model = TransformerModel(
            input_size=num_features,
            num_classes=num_classes,
            dropout=DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    model = model.to(device)

    print("\nModel Summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=MODEL_DIR,
        class_weights=class_weights,
        grad_clip=GRAD_CLIP_NORM,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
    )

    # Plot training history
    plot_training_history(history, MODEL_DIR)

    # Load best model
    model.load_state_dict(torch.load(Path(MODEL_DIR) / 'best_model.pth', weights_only=True))

    # Evaluate on test set
    results = evaluate_model(model, test_loader, class_names, device, MODEL_DIR)

    # Save final model
    model_path = Path(MODEL_DIR) / 'final_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved to {model_path}")

    # Save training results
    results_path = Path(MODEL_DIR) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model_type': MODEL_TYPE,
            'epochs_trained': len(history['train_loss']),
            'final_train_accuracy': float(history['train_acc'][-1]),
            'final_val_accuracy': float(history['val_acc'][-1]),
            'test_accuracy': float(results['test_accuracy']),
            'num_classes': num_classes,
            'class_names': class_names,
            'sequence_length': sequence_length,
            'features_per_frame': num_features,
            'uses_face_mesh': num_features in (186, 216),
            'uses_global_trajectory': num_features == 216,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }, f, indent=2)

    print(f"Results saved to {results_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nTest Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Model trained on: {device}")


if __name__ == "__main__":
    main()
    # Bypass Python's interpreter shutdown to avoid the
    # STATUS_STACK_BUFFER_OVERRUN (0xC0000409) crash that PyTorch + CUDA on
    # Windows triggers when cuDNN/CUDA DLLs unload in the wrong order. All
    # model artifacts and metrics are already written by main() above; we
    # only skip GC/atexit cleanup here.
    import os
    os._exit(0)
