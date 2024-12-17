import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, data_file):
        self.data = pd.read_excel(data_file)
        self.le_joke = LabelEncoder()
        self.le_reaction = LabelEncoder()
        self.imputer = SimpleImputer(strategy='most_frequent')

    def prepare_data(self):
        X = self.data['video_filename']
        y_joke = self.imputer.fit_transform(self.data['joke_type'].values.reshape(-1, 1)).ravel()
        y_reaction = self.imputer.fit_transform(self.data['reaction_type'].values.reshape(-1, 1)).ravel()

        y_joke_encoded = self.le_joke.fit_transform(y_joke)
        y_reaction_encoded = self.le_reaction.fit_transform(y_reaction)

        return train_test_split(X, y_joke_encoded, y_reaction_encoded, test_size=0.2, random_state=42)

    def get_num_classes(self):
        return len(self.le_joke.classes_), len(self.le_reaction.classes_)

    def get_class_names(self):
        return self.le_joke.classes_, self.le_reaction.classes_