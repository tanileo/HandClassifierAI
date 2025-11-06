import pandas as pd         # データ操作用ライブラリ
import numpy as np         # 数値計算用ライブラリ
import os                  # OS操作用ライブラリ
from glob import glob      # ファイルパス操作用ライブラリ

# 設定
input_dir = "hand_data"      # CSVフォルダ
output_path = "dataset.npy"  # 出力ファイル
skip_seconds = 5             # 最初の5秒をスキップ

# ランドマークデータを読み込み・前処理する関数
def load_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path)

    # 最初の5秒スキップ
    start_time = pd.to_datetime(df["timestamp"].iloc[0])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] > start_time + pd.Timedelta(seconds=skip_seconds)]

    # ランドマーク座標を取得（小文字に対応）
    coords = df.filter(regex="(x|y|z)\d+").values
    if coords.size == 0:
        raise ValueError(f"❌ 座標データが見つかりません: {file_path}")

    normalized_frames = []
    for frame in coords.reshape(-1, 21, 3):
        wrist = frame[0]
        rel = frame - wrist  # 手首原点化

        # 1フレームごとにスケール正規化
        scale = np.linalg.norm(rel, axis=1).max()
        if scale > 0:
            rel /= scale

        normalized_frames.append(rel.flatten())

    return np.array(normalized_frames)

# メイン処理
files = glob(os.path.join(input_dir, "*.csv"))
if not files:
    raise FileNotFoundError(f"❌ CSVファイルが見つかりません: {input_dir}")

data_by_class = {}
for f in files:
    class_name = os.path.splitext(os.path.basename(f))[0]  # Rock, Paper, Scissors
    coords = load_and_preprocess_csv(f)
    if class_name not in data_by_class:
        data_by_class[class_name] = []
    data_by_class[class_name].append(coords)

# クラス間でデータ数を揃える
min_len = min(min(len(c) for c in data_by_class[k]) for k in data_by_class)
print(f"📏 各クラス {min_len} サンプルに統一")

X, y = [], []
for class_name, datasets in data_by_class.items():
    class_data = np.vstack([c[:min_len] for c in datasets])
    X.append(class_data)
    y.append(np.full(len(class_data), class_name))

X = np.vstack(X)
y = np.concatenate(y)

np.save(output_path, {"X": X, "y": y})
print(f"✅ dataset saved to {output_path}")
print(f"   X shape = {X.shape}, y shape = {y.shape}")
print(f"   classes = {list(data_by_class.keys())}")
