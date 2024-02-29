from flask import Flask, request
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
from core.crop_cards import crop_single
from core.model import ConvolutionModel
import numpy as np

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['WTF_CSRF_ENABLED'] = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LABELS_DICT = {
    0: 'alb_id',
    1: 'aze_passport',
    2: 'esp_id',
    3: 'est_id',
    4: 'fin_id',
    5: 'grc_passport',
    6: 'lva_passport',
    7: 'rus_internalpassport',
    8: 'srb_passport',
    9: 'svk_id',
}

def pre_process_image(image: np.ndarray, is_cropped: bool, crop_image: bool) -> torch.Tensor:
    image = Image.fromarray(image)
    if not is_cropped and crop_image:
        image = crop_single(image)
    transformations = transforms.Compose([
        transforms.Pad(30),
        transforms.Resize((640, 480)),
        transforms.ToTensor()
    ])
    image: torch.Tensor = transformations(image)
    image = image.view(-1, *image.shape)
    return image.to(DEVICE)

def load_model(model_dir, width, height):
    model = ConvolutionModel(width, height, is_bw=False)
    model.load_state_dict(torch.load(model_dir))
    model.to(DEVICE)
    model.eval()
    return model

@app.route('/classify', methods=['POST'])
def classify():
    try:
        event = request.get_json()
        image = event['image']
        is_cropped = event['is_cropped']
        image = pre_process_image(image, is_cropped, True)
        model = load_model('models/cropped_best_model.pth', 640, 480)

        output = model(image)
        probability, predicted = torch.max(output, 1)
        return {
            'predicted': LABELS_DICT[predicted.item()],
            'probability': torch.exp(probability).item()
        }
    except Exception as e:
        return {
            'error': str(e)
        }
