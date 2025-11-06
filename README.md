# HandClassifierAI
## 実行順序
1. prepare_data.py
   Rock, Scissors, Paperで、それぞれいろいろな角度からデータを取る。
3. create_dataset.py
   Rock, Scissors, Paperのデータを合体してデータセットを作成
4. hand_train.py
   作成したデータセットを用いてニューラルネットワークで機械学習、モデル作成
5. hand_ai.py
   作成したモデルを使って、リアルタイムで手の形を識別
