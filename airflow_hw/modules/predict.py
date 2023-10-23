# <YOUR_IMPORTS>
import os
import dill
import glob
import json
import pandas as pd

from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')

test_json = glob.glob(f'{path}/data/test/*.json')


def predict():
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]

    with open(f'{path}/data/models/{latest_model}', 'rb') as cars_pipe:
        model = dill.load(cars_pipe)

    df_pred = pd.DataFrame(columns=['id', 'predictions'])

    for cars in test_json:
        with open(cars, 'r') as json_file:
            cars_dict = json.load(json_file)

        data = pd.DataFrame.from_dict([cars_dict])
        prediction = model.predict(data)

        pred_dict = {'id': data['id'].values[0], 'predictions': prediction[0]}
        df = pd.DataFrame([pred_dict])
        df_pred = pd.concat([df, df_pred], ignore_index=True)

    df_pred.to_csv(f'{path}/data/predictions/predict_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
