import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import ConvolutionModel
from crop_cards import crop_single

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

def pre_process_image(image: str, is_cropped: bool, crop_image: bool) -> torch.Tensor:
    image = Image.open(image)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path',dest='image_path', type=str, help='Path to image to predict')
    parser.add_argument('--model_path',dest='model_path', type=str, help='Path to the model to use')
    parser.add_argument('--width',dest='width', type=int, help='Width of the image', default=640)
    parser.add_argument('--height',dest='height', type=int, help='Height of the image', default=480)
    parser.add_argument('--is_cropped',dest='is_cropped', type=bool, help='Whether the image is precropped', default=False)
    parser.add_argument('--crop',dest='crop', type=bool, help='Whether to crop the image', default=False)
    args = parser.parse_args()
    
    image = pre_process_image(args.image_path, args.is_cropped, args.crop)
    model = load_model(args.model_path, args.width, args.height)
    output = model(image)
    probability, predicted = torch.max(output, 1)
    print(f" Predicted Class: {LABELS_DICT[predicted.item()]} with Probability: {torch.exp(probability).item()}")
