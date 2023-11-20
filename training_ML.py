import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


all_data = pd.read_csv('asos_power.csv')
relevant_loc = ['Youngam']
all_data = all_data[all_data['Name'].isin(relevant_loc)]
all_data.fillna(0.0001, inplace = True)

train = all_data[all_data['year'].isin([2014, 2015, 2016, 2017, 2018, 2019, 2020])]
test = all_data[all_data['year'].isin([2021, 2022])]

X_train = train.drop(columns = ['Location', 'Name', 'Date', 'day', 'pve'], axis = 1).values
y_train = train['pve'].values
X_test = test.drop(columns = ['Location', 'Name', 'Date', 'day', 'pve'], axis = 1).values
y_test = test['pve'].values
test_dates = test['Date']

def initialize_models():
    models = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0),
        "XGBoost": XGBRegressor(use_label_encoder = False, eval_metric = 'logloss'), 
        "NGBoost":NGBRegressor()
    }
    return models

def predict(models, X_train, y_train, X_test, y_test, test):
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        predictions = pd.DataFrame(predictions)
        predictions['Actual'] = y_test
        predictions = predictions.set_index(test.index)
        predictions['Date'] = test['Date']

    return predictions

def plot(df, option):
    df['Date'] = pd.to_datetime(df['Date'])

    if option == 2: #daily sum
        df = df.groupby(df['Date'].dt.date).sum()

    models = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 'Extra Trees', 'CatBoost', 'XGBoost', 'NGBoost']
    
    for model in models:
        plt.figure(figsize=(12,6))

        #actual vs. predict
        plt.subplot(1,2,1)
        plt.plot(df['Actual'], label='Actual', color='blue')
        plt.plot(df[model], label='Prediction', color='red')
        plt.title(f'{model} - Actual vs. Prediction')
        plt.legend()

        #predict - actual
        plt.subplot(1, 2, 2)
        error = df[model] - df['Actual']
        plt.plot(error, color='purple')
        plt.title(f'{model} - Prediction Error (Prediction - Actual)')
        plt.axhline(y=0, color='gray', linestyle='--')

        plt.tight_layout()
        plt.show()

def calculate_mae(df):
    mae_values = {}
    model_names = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 'Extra Trees', 'CatBoost', 'XGBoost', 'NGBoost']

    for model in model_names:
        mae = mean_absolute_error(df['Actual'], df[model])
        mae_values[model] = mae
    for model, mae in mae_values.items():
        print(f'{model} : {mae}')

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
models = initialize_models()
predictions = predict(models, X_train, y_train, X_test ,y_test, test)
calculate_mae(predictions)
#plot(predictions, 2)

