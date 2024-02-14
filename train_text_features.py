import pandas as pd
import re 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

FEATURES = {
    'SHQIPERISE': 'ALBANIA',
    'SHQIPÈRISE': 'ALBANIA',
    'SHOIPÊRISE': 'ALBANIA',
    'SHOIPÊRISÉ': 'ALBANIA',
    'SHQIPÊRISÉ': 'ALBANIA',
    'SHOIPÉRISÉ': 'ALBANIA',
    'SHOIPERISÉ': 'ALBANIA',
    'SHQIPËRISE': 'ALBANIA',
    '<': 'PASSPORT',
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
            # Replace < with PASSPORT 
            ocr_data.at[idx, 'text'] = ocr_data.at[idx, 'text'].replace(feature, FEATURES[feature])

            # Remove special characters and numbers but retain accented characters
            ocr_data.at[idx, 'text'] = re.sub(r"[-!$%^&*()_+|~=`{}\[\]:\";'<>?,.\/0-9]", ' ', ocr_data.at[idx, 'text'])
            
            # Combine ocr in a single line
            ocr_data.at[idx, 'text'] = " ".join(ocr_data.at[idx, 'text'].split('\n'))
        
        # Set the label for the ocr text
        ocr_data.at[idx, 'label'] = LABELS_DICT[ocr_data.at[idx, 'class']]
    return ocr_data

def train(cleaned_data: pd.DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(cleaned_data['text'], cleaned_data['label'], test_size=0.4, random_state=42)

    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, max_features=500)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model = model.fit(x_train, y_train)

    print(f"Accuracy Score: {accuracy_score(y_test, model.predict(x_test))}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, model.predict(x_test))}")
    return model, vectorizer

def save_models(model, vectorizer):
    with open('models/logistic_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('models/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

if __name__ == '__main__':
    ocr_data = pd.read_csv('csv/ocr.csv') # Read raw ocr data
    cleaned_data = pre_process_text(ocr_data) # Pre-process the ocr data
    cleaned_data.to_csv('csv/ocr_cleaned.csv', index=False) # Save the cleaned ocr data
    model, vectorizer = train(cleaned_data) # Train the model
    save_models(model, vectorizer) # Save the models
    print('Models trained and  saved')