from ultralytics import YOLO

# YOLOモデルをロードまたは初期化
model = YOLO("yolo11n.pt")  # カスタムモデル構成ファイル (.yaml)

# 学習を実行
if __name__ == '__main__':
    model.train(
        data="C:\\Users\\kyuta\\programing\\jikken\\data.yaml",  # data.yamlファイルの正確なパス
        epochs=100,                     # エポック数
        batch=16,                       # バッチサイズ
        imgsz=1024,                     # 画像サイズ (大きめに設定)
        device=0,                       # GPUを指定 (例: 0 = GPUのインデックス)
        rect=True                       # アスペクト比を保持したリサイズ
    )