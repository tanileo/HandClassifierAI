import cv2                      # 画像処理用のライブラリ
import mediapipe as mp          # 手の検出とランドマーク推定用のライブラリ
import csv                      # CSV操作用ライブラリ
import numpy as np              # 数値計算用ライブラリ
from datetime import datetime   # 日時取得用ライブラリ

# 設定
CLASS_NAME = input("保存するクラス名（例: Rock, Scissors, Paper）: ")
CSV_PATH = f"./hand_data/{CLASS_NAME}.csv"

mp_hands = mp.solutions.hands                                            # 手検出モジュールの初期化
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)    # 手検出モデルの初期化（認識する手の数=1、最小検出信頼度=0.5）

# CSV初期化
with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ["timestamp"] + [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']]
    writer.writerow(header)


# メインループ
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:                                      # 手が検出された場合
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # === 手首基準の相対座標化 ===
            wrist = landmarks[0]
            rel_landmarks = landmarks - wrist

            # === スケーリング（正規化） ===
            max_range = np.max(np.linalg.norm(rel_landmarks, axis=1))
            if max_range > 0:
                rel_landmarks /= max_range

            # === 保存 ===
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row = [now] + rel_landmarks.flatten().tolist()

            with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    cv2.imshow("Collect Hand Data", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"データ収集終了: {CSV_PATH}")
