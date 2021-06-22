import numpy as np
import pandas as pd
import plotly.graph_objects as go


def get_data():
    data = pd.read_csv('face_age_dataset/age_gender.csv')

    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

    data.head()

    print('Total rows: {}'.format(len(data)))
    print('Total columns: {}'.format(len(data.columns)))

    data['pixels'] = data['pixels'].apply(lambda x: x / 255)

    age_dist = data['age'].value_counts()
    gender_dist = data['gender'].value_counts().rename(index={0: 'Male', 1: 'Female'})

    def ditribution_plot(x, y, name):
        fig = go.Figure([
            go.Bar(x=x, y=y)
        ])

        fig.update_layout(title_text=name)
        fig.show()

    print(ditribution_plot(x=age_dist.index, y=age_dist.values, name='Age Distribution'))
    print(ditribution_plot(x=gender_dist.index, y=gender_dist.values, name='Gender Distribution'))
