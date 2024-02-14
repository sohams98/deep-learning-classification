from torch.utils.data import DataLoader
from torch import optim, Tensor
from torch import nn
import torch
import tqdm
from model import ConvolutionModel, ImageDataset

MODEL_NAME = 'cropped_bw'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data_loaders(data_dir, width, height, is_bw):
    train_data = ImageDataset(f'{data_dir}/train', width, height, is_bw=is_bw)
    val_data = ImageDataset(f'{data_dir}/val', width, height, is_bw=is_bw)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)
    return train_loader, val_loader

def load_model(width, height, is_bw):
    model = ConvolutionModel(width, height, is_bw=is_bw)
    model.to(DEVICE)
    return model

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    best_model = {
        'accuracy': 0,
        'epoch': 0,
        'model': None,
        'optimizer': None
    }
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(20):
        for image, label in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
            image: Tensor = image.to(DEVICE)
            label: Tensor = label.to(DEVICE)
            y_pred: Tensor = model(image)
            
            optimizer.zero_grad()
            loss = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()
        
        acc = 0
        count = 0
        for inputs, labels in tqdm.tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            y_pred = model(inputs)

            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
        acc /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
        torch.save(model.state_dict(), f'models/{MODEL_NAME}.pth')
        if acc > best_model['accuracy']:
            best_model['accuracy'] = acc
            best_model['epoch'] = epoch
            best_model['model'] = model.state_dict()
            best_model['optimizer'] = optimizer.state_dict()

    torch.save(best_model, f'models/{MODEL_NAME}_best_model_parameters.pth')
    torch.save(model.state_dict(), f'models/{MODEL_NAME}_best_model.pth')

if __name__ == '__main__':
    data_dir = 'datasets/classify_cropped'
    width, height = 640, 480
    is_bw = False
    train_loader, val_loader = get_data_loaders(data_dir, width, height, is_bw)
    model = load_model(width, height, is_bw)
    train(model, train_loader, val_loader)
