import numpy as np
import pandas as pd

data = pd.read_csv(r'dataset/age_gender.csv')

data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

data.head()

print('Total rows: {}'.format(len(data)))
print('Total columns: {}'.format(len(data.columns)))

# normalizing pixels data
data['pixels'] = data['pixels'].apply(lambda x: x / 255)

# calculating distributions
age_dist = data['age'].value_counts()
# ethnicity_dist = data['ethnicity'].value_counts()
gender_dist = data['gender'].value_counts().rename(index={0: 'Male', 1: 'Female'})

# def ditribution_plot(x, y, name):
#     fig = go.Figure([
#         go.Bar(x=x, y=y)
#     ])
#
#     fig.update_layout(title_text=name)
#     fig.show()


# ditribution_plot(x=age_dist.index, y=age_dist.values, name='Age Distribution')

X = np.array(data['pixels'].tolist())

## Converting pixels from 1D to 3D
print(X.shape)
X = X.reshape(X.shape[0], 48, 48, 1)
