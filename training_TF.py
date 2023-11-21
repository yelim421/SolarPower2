import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tensorflow_probability as tfp

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


all_data = pd.read_csv('asos_power.csv')
relevant_loc = ['Youngam']
all_data = all_data[all_data['Name'].isin(relevant_loc)]
all_data.fillna(0.0001, inplace = True)

train = all_data[all_data['year'].isin([2014, 2015, 2016, 2017, 2018, 2019, 2020])]
test = all_data[all_data['year'].isin([2021, 2022])]

X = train.drop(columns = ['Location', 'Name', 'Date', 'day', 'pve'], axis = 1).values
y = train['pve'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
X_test = test.drop(columns = ['Location', 'Name', 'Date', 'day', 'pve'], axis = 1).values
y_test = test['pve'].values

model_pre = LGBMRegressor()
model_pre.fit(X, y)
X_new = X.copy()
X_new['pve_pred'] = model_pre.predict(X)

def loss_fn(y_true, y_pred):
    return tfp.stats.percentile(tf.abs(y_true - y_pred), q=50)

def metric_fn(y_true, y_pred):
    return tfp.stats.percentile(tf.abs(y_true - y_pred), q=100) - tfp.stats.percentile(tf.abs(y_true - y_pred), q=0)


callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='min',restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.00001),
    tf.keras.callbacks.TerminateOnNaN()
]

def create_model():

    input_layer = tf.keras.Input(shape=(len(features), ))
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(input_layer)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    #x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    #x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(x)
    output_layer = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.013, beta_1=0.5),
                  loss=loss_fn,
                  metrics=metric_fn)

    return model

model = create_model()
history = model.fit(X_new.astype('float32'), y.astype('float32'),
                    epochs=100,
                    class_weight=model_pre.class_weight,
                    callbacks=callbacks_list,
                    validation_split=0.1)

