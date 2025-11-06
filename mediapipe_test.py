# 使用するライブラリのインポート
import cv2                 # 画像処理用のライブラリ
import mediapipe as mp     # 手の検出とランドマーク推定用のライブラリ

# 設定
mp_hands = mp.solutions.hands                                           # 手検出モジュールの初期化
mp_drawing = mp.solutions.drawing_utils                                 # ランドマーク描画用ユーティリティ
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)   # 手検出モデルの初期化（認識する手の数=1、最小検出信頼度=0.5）

# メインループ
cap = cv2.VideoCapture(0)

# 無限ループでカメラ映像を処理
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:                                                      # 手が検出された場合
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)   # ランドマークと接続線を描画

    cv2.imshow("mediapipe test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()
