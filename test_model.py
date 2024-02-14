import argparse
import torch
from torch.utils.data import DataLoader
from model import ConvolutionModel, ImageDataset, ImageDatasetPadded

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_test_loader(data_path: str, width: int, height: int, is_bw: bool, pad: bool):
    test_data = ImageDataset(f'{data_path}/test', width, height, is_bw=is_bw) if not pad else ImageDatasetPadded(f'{data_path}/test', width, height, is_bw=is_bw)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=True, pin_memory=True, pin_memory_device=DEVICE)
    return test_loader

def load_model(model_path: str, width: int, height: int, is_bw: bool):
    model = ConvolutionModel(width, height, is_bw=is_bw)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    return model

def test(model: ConvolutionModel, test_loader: DataLoader):
    acc = 0
    count = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        y_pred = model(inputs)

        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Model accuracy %.2f%%" % (acc*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--model_path',dest='model_path', type=str, help='Path to the model to use')
    parser.add_argument('--data_path', dest='data_path', type=str, help='Path to the data directory')
    parser.add_argument('--width', dest='width', type=int, help='Width of the image', default=640)
    parser.add_argument('--height', dest='height', type=int, help='Height of the image', default=480)
    parser.add_argument('--is_bw', dest='is_bw', type=bool, help='Is the image black and white', default=False)
    parser.add_argument('--pad', dest='pad', type=bool, help='Should the image be padded', default=False)
    args = parser.parse_args()

    model_path = args.model_path
    test_dir = args.data_path
    width, height = args.width, args.height
    is_bw = args.is_bw
    pad = args.pad
    test_loader = get_test_loader(test_dir, width, height, is_bw, pad)
    model = load_model(model_path, width,height, is_bw)
    test(model, test_loader)