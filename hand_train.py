import torch                                                              # PyTorchライブラリ
import torch.nn as nn                                                     # ニューラルネットワーク用ライブラリ
import torch.optim as optim                                               # 最適化アルゴリズム用ライブラリ
from torch.utils.data import TensorDataset, DataLoader, random_split      # データセット・データローダー用ライブラリ
import numpy as np                                                        # 数値計算用ライブラリ

# データ読み込み
data = np.load("dataset.npy", allow_pickle=True).item()
X, y = data["X"], data["y"]

# ラベルを数値に変換
labels = sorted(set(y))
label_to_idx = {l: i for i, l in enumerate(labels)}
idx_to_label = {i: l for l, i in label_to_idx.items()}
y = np.array([label_to_idx[v] for v in y])

# torch化
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X, y)

# データ分割
train_size = int(len(dataset)*0.8)
train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# モデル定義
class HandClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.net(x)

input_size = X.shape[1]
num_classes = len(labels)
model = HandClassifier(input_size, 128, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 学習
prev_loss = None
for epoch in range(300):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # 学習率調整
    if prev_loss is not None:
        if abs(prev_loss - avg_loss) < 1e-5 or avg_loss > prev_loss:
            lr = optimizer.param_groups[0]['lr'] * 0.5
            optimizer.param_groups[0]['lr'] = max(lr, 1e-6)
    prev_loss = avg_loss

    # 検証
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: loss={avg_loss:.6f}, val_acc={acc:.4f}")

# モデル保存
torch.save({
    "model_state_dict": model.state_dict(),
    "label_to_idx": label_to_idx,
    "idx_to_label": idx_to_label
}, "hand_classifier.pth")
print("✅ モデルを hand_classifier.pth に保存しました")
