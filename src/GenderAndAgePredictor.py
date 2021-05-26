import os
import cv2
from tensorflow.keras.models import load_model
from loader import load_images_from_folder


class GenderAndAgePredictor:

    def __init__(self, gender_model_path=None, age_model_path=None):
        self.gender_model = self.get_model_from_path(gender_model_path)
        self.age_model = self.get_model_from_path(age_model_path)

    def get_model_from_path(self, model_path):
        model = load_model(os.path.abspath(model_path))
        return model

    def predict_gender_and_age(self, image):
        print(self.gender_model.predict(image))


if __name__ == '__main__':

    images = load_images_from_folder("input")
    first_image = images[0]
    first_image = cv2.resize(first_image, (48, 48))
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Display window", first_image)
    # k = cv2.waitKey(0)

    predictor = GenderAndAgePredictor()
    predictor.predict_gender_and_age(first_image)
