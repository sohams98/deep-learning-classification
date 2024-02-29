import glob
from typing import List
import cv2
import os
import tqdm
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results 
from PIL import Image
import torch

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
def crop():
    model = YOLO('card/card/weights/best.pt')

    for image in tqdm.tqdm(glob.glob('./datasets/images/*/*')):
        results: List[Results] = model.predict(image, device='0', conf=0.25, save_conf=True)
        for r in results:
            box: Boxes  = r.boxes

            coordinates = box.xyxy[0].cpu().numpy()
            x1 = coordinates[0]
            y1 = coordinates[1]
            x2 = coordinates[2]
            y2 = coordinates[3]

            img = cv2.imread(image)
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]
            file_name = image.split('\\')[-1]
            folder_name = image.split('\\')[-2]

            if not os.path.exists(f'./datasets/cropped/{folder_name}'):
                os.makedirs(f'./datasets/cropped/{folder_name}', exist_ok=True)
            cv2.imwrite(f'./datasets/cropped/{folder_name}/{file_name}', crop_img)

def crop_single(image: Image):
    model = YOLO('./models/best.pt')

    results: List[Results] = model.predict(image, device=DEVICE, conf=0.25, save_conf=True)
    crop_img = np.asarray(image)

    box: Boxes  = results[0].boxes

    coordinates = box.xyxy[0].cpu().numpy()
    x1 = coordinates[0]
    y1 = coordinates[1]
    x2 = coordinates[2]
    y2 = coordinates[3]

    crop_img = crop_img[int(y1):int(y2), int(x1):int(x2)]
    return Image.fromarray(crop_img)

if __name__ == '__main__':
    crop()
