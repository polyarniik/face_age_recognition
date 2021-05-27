import cv2
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from loader import load_images_from_folder
from tensorflow.keras.preprocessing.image import img_to_array
from GenderAndAgePredictor import GenderAndAgePredictor

if __name__ == '__main__':
    images = load_images_from_folder("input")

    predictor = GenderAndAgePredictor(
        gender_model_path='/home/pain/Desktop/sem_works/face_age_recognition/gender_recognise.model',
        age_model_path='/home/pain/Desktop/sem_works/face_age_recognition/age_recognise.model'
    )
    for first_image in images:

        # first_image = images[0]
        first_image = cv2.resize(first_image, (48, 48))
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)

        data = ''
        for i in range(48):
            for j in range(48):
                data += " " + str(first_image[i][j])

        x = data[1:]
        first_image = np.array(x.split(), dtype="float32") / 255
        X = first_image.reshape(1,48,48,1)

        # # cv2.imshow("Display window", first_image)
        # # k = cv2.waitKey(0)
        predictor.predict_gender_and_age(X)
        print()
