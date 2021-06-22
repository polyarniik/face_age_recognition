import os

from tensorflow.keras.models import load_model

GENDER = {0: 'Male', 1: 'Female'}


class GenderAndAgePredictor:

    def __init__(self, gender_model_path, age_model_path):
        self.gender_model = self.get_model_from_path(gender_model_path)
        self.age_model = self.get_model_from_path(age_model_path)

    @staticmethod
    def get_model_from_path(model_path):
        model = load_model(os.path.abspath(model_path))
        return model

    def predict_gender_and_age(self, image):
        gender_predict = int(self.gender_model.predict_classes(image))
        return GENDER[gender_predict], int(self.age_model.predict(image))
