import os
import cv2
from tensorflow.keras.models import load_model
from loader import load_images_from_folder


class GenderAndAgePredictor:

    def __init__(self, gender_model_path, age_model_path=None):
        self.gender_model = self.get_model_from_path(gender_model_path)
        self.age_model = self.get_model_from_path(age_model_path)

    def get_model_from_path(self, model_path):
        model = load_model(os.path.abspath(model_path))
        return model

    def predict_gender_and_age(self, image):
        gender = {0: 'Male', 1: 'Female'}
        gender_predict = int(self.gender_model.predict_classes(image))
        print(gender[gender_predict])
        print(self.age_model.predict(image))


