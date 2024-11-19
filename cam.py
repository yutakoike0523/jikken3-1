import cv2
from ultralytics import YOLO

# YOLOモデルの読み込み
model = YOLO('runs/detect/train5/weights/best.pt')  # トレーニング済みモデルを指定

# Webカメラを設定
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラを指定

# カメラが開けない場合のエラーチェック
if not cap.isOpened():
    print("Webカメラを開けませんでした。")
    exit()

while True:
    # フレームを取得
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした。")
        break

    # モデルで推論
    results = model(frame)

    # 検出結果をフレームに描画
    annotated_frame = results[0].plot()

    # フレームを表示
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
