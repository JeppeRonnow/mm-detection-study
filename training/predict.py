from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../runs/detect/mm_detector3/weights/best.pt")

    results = model.predict('../dataset/test/images/*.jpg', save=True, conf=0.5)

