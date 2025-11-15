from ultralytics import YOLO

if __name__ == '__main__':
    # Load trained model

    model = YOLO("../runs/detect/mm_detector3/weights/best.pt")

    metrics = model.val()

    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
