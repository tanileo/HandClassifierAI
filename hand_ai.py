import cv2                     # 画像処理用のライブラリ
import mediapipe as mp         # 手の検出とランドマーク推定用のライブラリ
import torch                   # PyTorchライブラリ
import torch.nn as nn          # ニューラルネットワーク用ライブラリ
import numpy as np             # 数値計算用ライブラリ


# モデル定義（学習時と同じ構造）
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

# モデルロード
checkpoint = torch.load("hand_classifier.pth", map_location="cpu", weights_only=False)   # 学習済みモデル読み込み
idx_to_label = checkpoint["idx_to_label"]
input_size = 63        # 21ランドマーク×3座標
num_classes = len(idx_to_label)   # 3クラス（グー、チョキ、パー）

model = HandClassifier(input_size, 128, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Mediapipe設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ランドマーク取得
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            wrist = coords[0]
            rel = coords - wrist

            # 左手なら反転
            if hand_landmarks.landmark[17].x < hand_landmarks.landmark[5].x:
                rel[:, 0] *= -1

            # 正規化
            scale = np.linalg.norm(rel, axis=1).max()
            if scale > 0:
                rel /= scale

            # flatten
            inp = torch.tensor(rel.flatten(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = model(inp)
                label_idx = torch.argmax(pred, dim=1).item()
                label = idx_to_label[label_idx]

            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
