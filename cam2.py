from ultralytics import YOLO
import cv2

# YOLO11nモデルをロード
model = YOLO("yolo11n.pt")

# カメラの映像を取得
cap = cv2.VideoCapture(0)  # '0' はデフォルトのカメラを指定。複数カメラがある場合は番号を変更。

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # カメラからフレームを取得
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # モデルを使って推論
    results = model.predict(frame, imgsz=640, conf=0.5, show=False)

    # 推論結果を取得して描画
    annotated_frame = results[0].plot()  # 検出された物体をフレームに描画

    # 結果を画面に表示
    cv2.imshow("YOLOv11 Object Detection", annotated_frame)

    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
