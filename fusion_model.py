import pickle
import torch 
from torch import Tensor
from torchvision import transforms
from PIL import Image
import pandas as pd
import glob
import json 
import tqdm 
import numpy as np
import re
from model import ConvolutionModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURES = {
    'SHQIPERISE': 'ALBANIA',
    'SHQIPÈRISE': 'ALBANIA',
    'SHOIPÊRISE': 'ALBANIA',
    'SHOIPÊRISÉ': 'ALBANIA',
    'SHQIPÊRISÉ': 'ALBANIA',
    'SHOIPÉRISÉ': 'ALBANIA',
    'SHOIPERISÉ': 'ALBANIA',
    'SHQIPËRISE': 'ALBANIA',
    '<<': 'PASSPORT',
    'LETERNJOFTIM': 'IDENTITY',
    'LETERNIOFTIL': 'IDENTITY',
    'LETERNIOFTIM': 'IDENTITY',
    'NACIONALDE': 'NACIONAL DE',
    'DEIDENTIDAD': 'DE IDENTIDAD',
}

LABELS_DICT = {
    'alb_id':0,
    'aze_passport':1,
    'esp_id':2,
    'est_id':3,
    'fin_id':4,
    'grc_passport':5,
    'lva_passport':6,
    'rus_internalpassport':7,
    'srb_passport':8,
    'svk_id':9,
}

def pre_process_text(ocr_data: pd.DataFrame):
    for idx, row in ocr_data.iterrows():
        for feature in FEATURES:
            # Replace the potentially incorrect ocr text with the correct feature
            # Replace << with PASSPORT 
            ocr_data.at[idx, 'text'] = ocr_data.at[idx, 'text'].replace(feature, f" {FEATURES[feature]} ")

            # Remove special characters and numbers but retain accented characters
            ocr_data.at[idx, 'text'] = re.sub(r"[-!$%^&*()_+|~=`{}\[\]:\";'<>?,.\/0-9]", ' ', ocr_data.at[idx, 'text'])
            
            # Combine ocr in a single line
            ocr_data.at[idx, 'text'] = " ".join(ocr_data.at[idx, 'text'].split('\n'))
        
        # Set the label for the ocr text
        ocr_data.at[idx, 'label'] = LABELS_DICT[ocr_data.at[idx, 'class']]
    return ocr_data

def pre_process_image(image: str) -> torch.Tensor:
    image = Image.open(image)
    transformations = transforms.Compose([
        transforms.Pad(30),
        transforms.Resize((640, 480)),
        transforms.ToTensor()
    ])
    image: torch.Tensor = transformations(image)
    return image.view(-1, *image.shape)

def load_models():
    # Load the text models
    vectorizer: TfidfVectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    text_features_model: LogisticRegression = pickle.load(open('models/logistic_model.pkl', 'rb'))
    
    # Load the PyTorch model
    model = ConvolutionModel()
    model.load_state_dict(torch.load('models/cropped_best_model.pth'))
    model.to('cuda')
    model.eval()
    return model, text_features_model, vectorizer

def test(image: str, ocr_data: pd.DataFrame, model: ConvolutionModel, text_features_model: LogisticRegression, vectorizer: TfidfVectorizer):
    df = {
        'actual_class': '',
        'tf_predicted_class': '',
        'tf_predicted_prob': '',
        'tf_match': '',
        'tf_predicted_features': '',
        'cnn_predicted_class': '',
        'cnn_predicted_prob': '',
        'cnn_match': '',
        'cnn_predicted_features': '',
        'fusion_predicted_class': '',
        'fusion_predicted_prob': '',
        'fusion_match': '',
    }

    image_name = image.split('\\')[-1]
    class_name = image.split('\\')[-3]

    ocr = ocr_data[(ocr_data['class'] == class_name) & (ocr_data['image'] == image_name)]

    test_image: Tensor = pre_process_image(image)
    test_image = test_image.to('cuda')

    vectorized_text = vectorizer.transform(ocr['text'])

    text_features = text_features_model.predict(vectorized_text)
    text_features_prob: np.ndarray = text_features_model.predict_proba(vectorized_text)
    
    output = model(test_image)
    _, predicted = torch.max(output, 1)
    
    tf_match = 1 if text_features[0] == LABELS_DICT[class_name] else 0
    cnn_match = 1 if predicted.item() == LABELS_DICT[class_name] else 0
    df['actual_class'] = LABELS_DICT[class_name]
    df['tf_predicted_class'] = text_features[0]
    df['tf_match'] = tf_match
    df['tf_predicted_prob'] = np.sort(text_features_prob[0])[-1]
    df['tf_predicted_features'] = json.dumps(text_features_prob[0].tolist())
    df['cnn_predicted_class'] = predicted.item()
    df['cnn_match'] = cnn_match
    df['cnn_predicted_prob'] = torch.exp(_).item()
    df['cnn_predicted_features'] = json.dumps(torch.exp(output).tolist()[0])

    fusion_probs = text_features_prob[0]*0.5 + torch.exp(output).cpu().detach().numpy()[0]*0.5
    fusion_predicted = np.argmax(fusion_probs)
    fusion_match = 1 if fusion_predicted == LABELS_DICT[class_name] else 0
    df['fusion_predicted_class'] = fusion_predicted
    df['fusion_match'] = fusion_match
    df['fusion_predicted_prob'] = np.sort(fusion_probs)[-1]

    return pd.DataFrame(df, index=[0])
    

if __name__ == '__main__':
    test_dir = 'datasets/cropped/test'
    ocr_data = pd.read_csv('csv/ocr_cleaned.csv')
    
    # optional step to preprocess text if not already done
    # ocr_data = pre_process_text(ocr_data)

    model, text_features_model, vectorizer = load_models()
    
    dfs = []
    for image in tqdm.tqdm(glob.glob(f'{test_dir}\\*\\images\\*.jpg', recursive=True)):
        df = test(image, ocr_data, model, text_features_model, vectorizer)
        dfs.append(df)
    final_results = pd.concat(dfs, ignore_index=True)
    final_results.to_csv('csv/results_fusion.csv', index=False)