from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')

    model.train(data='./card.yml', epochs=100,imgsz=640, project='card', name='card', device='0', exist_ok=True)
    # model.save('yolov8n_card_seg.pt')