import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from config import DATA_FILE
from feature_extraction import extract_all_features

def load_and_preprocess_data():
    data = pd.read_excel(DATA_FILE)
    X = data['video_filename']
    y_joke = data['joke_type']
    y_reaction = data['reaction_type']

    imputer = SimpleImputer(strategy='most_frequent')
    y_joke = imputer.fit_transform(y_joke.values.reshape(-1, 1)).ravel()
    y_reaction = imputer.fit_transform(y_reaction.values.reshape(-1, 1)).ravel()

    le_joke = LabelEncoder()
    le_reaction = LabelEncoder()
    y_joke_encoded = le_joke.fit_transform(y_joke)
    y_reaction_encoded = le_reaction.fit_transform(y_reaction)

    return train_test_split(X, y_joke_encoded, y_reaction_encoded, test_size=0.2, random_state=42)

def extract_features(X):
    features = []
    for video_file in X:
        feat = extract_all_features(video_file)
        if feat is not None:
            features.append(feat)
    features_array = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)
    return scaled_features