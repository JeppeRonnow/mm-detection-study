from ultralytics import YOLO


if __name__ == '__main__':
    # Load pretrained model
    model = YOLO('yolo11s.pt')

    # path to data.yaml
    data_yaml = "../dataset/data.yaml"

    # Train
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        name='mm_detector'
    )

    print(f"Best model gemt i: {model.trainer.best}")
