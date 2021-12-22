import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
import os
import sys
from io import StringIO
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()
current_path = str(current_path)
current_path = current_path.replace("\\","/")

st.title("Weather forecast")
st.sidebar.title("What to do")
dataset = st.sidebar.selectbox(label="Select a dataset", index=0, options=["jena_climate_2009_2016"])
activities = ["Exploratory Data Analysis", "Plotting and Visualization", "Building Model & Testing", "Forecasting using Model", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)
if dataset == "jena_climate_2009_2016":
    import zipfile
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = tensorflow.keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    zip_file = zipfile.ZipFile(zip_path)
    zip_file.extractall()
    csv_path = "jena_climate_2009_2016.csv"
    df = pd.read_csv(csv_path)

def resample():
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df.set_index('Date Time', inplace=True)
    data = df.resample('60T').mean()
    return data

data = resample()

if choice == "Exploratory Data Analysis":
    if st.checkbox("Show Dataset"):
        st.dataframe(df.head())
    if st.checkbox("Show columns"):
        st.write(df.columns)
    if st.checkbox("Show shape"):
        st.write(df.shape)
    if st.checkbox("Summary of DataSet"):
        st.write(df.describe())
    if st.checkbox("Value Counts"):
        st.write(df.count())
    if st.checkbox("Show Dataset after resampling"):
        st.dataframe(data.head())
    if st.checkbox("Show shape after resampling"):
        st.write(data.shape)
    if st.checkbox("Value Counts after resampling"):
        st.write(data.count())


def interpolate(data):
    data = data[['T (degC)']]
    data['T (degC)'] = data['T (degC)'].interpolate()
    return data
data = interpolate(data)

if choice == "Plotting and Visualization":
    all_columns = df.columns.tolist()
    if st.checkbox("Temperature Plot"):
        st.title('Temperature Series')
        st.line_chart(data)
    if st.checkbox("Correlation Heatmap"):
        st.write(sns.heatmap(df.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))
        st.pyplot()

def scaling_data(data):
    data = data.values
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sc = scaler.fit_transform(data)
    return data, scaler, sc
data, scaler, sc = scaling_data(data)

def split_data(data, sc):
    timestep = 36

    X = []
    Y = []

    for i in range(len(sc) - (timestep)):
        X.append(sc[i:i + timestep])
        Y.append(sc[i + timestep])

    X = np.asanyarray(X)
    Y = np.asanyarray(Y)

    k = 69000
    Xtrain = X[:k, :, :]
    Xtest = X[k:, :, :]
    Ytrain = Y[:k]
    Ytest = Y[k:]

    return Xtrain, Xtest, Ytrain, Ytest

Xtrain, Xtest, Ytrain, Ytest = split_data(data=data, sc=sc)



# THIS NEXT COMMENTED BLOCKS OF CODE IS FOR MODEL BUILDING AND FITTING BUT SINCE IT TAKES TIME FOR THE MODEL TO RUN,
# WE HAVE ALREADY RUN IT AND LOADED THE SAVED MODEL.

# model = Sequential()
# model.add(LSTM(32,activation = 'relu', input_shape= (36,1), return_sequences=True))
# model.add(LSTM(32, activation='relu', return_sequences=True))
# model.add(LSTM(32, activation='sigmoid', return_sequences=False))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# history = model.fit(Xtrain,Ytrain,epochs=15, verbose=1, callbacks=[callback])


model = tensorflow.keras.models.load_model(current_path+"/model.h5")

# Prediction for Xtest and then inverse transforming for plotting Ytest and Predicted values
preds= model.predict(Xtest)
preds = scaler.inverse_transform(preds)

Ytest=np.asanyarray(Ytest)
Ytest=Ytest.reshape(-1,1)
Ytest = scaler.inverse_transform(Ytest)

Ytrain=np.asanyarray(Ytrain)
Ytrain=Ytrain.reshape(-1,1)
Ytrain = scaler.inverse_transform(Ytrain)

test_df = pd.DataFrame(Ytest,columns=['Actual'])
pred_df = pd.DataFrame(preds,columns=['Predicted'])
concat_table = pd.concat([test_df,pred_df],axis=1)

test = Ytest.flatten()
pred = preds.flatten()

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

if choice == "Building Model & Testing":
    if st.checkbox("Model Summary"):
        model.summary()
    if st.checkbox("(MSE) Ytest VS Predicted_Ytest"):
        st.write(mean_squared_error(Ytest,preds))
    if st.checkbox("Plot Ytest VS Predicted_Ytest"):
        plt.rcParams.update({'font.size': 20})
        fig, ax= plt.subplots(1,1, figsize=(20,9))
        ax.plot(range(1045,1093),test[1045:], 'b', label="Test")
        ax.plot(range(1045,1093),pred[1045:], 'r', label="Predicted")
        ax.set_xlabel('Final 48 hours observations of test data')
        ax.set_ylabel('Temperature')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        st.pyplot(fig)
    if st.checkbox("Values Ytest VS Predicted_Ytest"):
        st.write(concat_table)

sys.stdout = old_stdout
st.text(mystdout.getvalue())

def insert_end(Xin,new_input):
    timestep = 36
    for i in range(timestep-1):
        Xin[:, i, :] = Xin[:, i+1, :]
    Xin[:, timestep-1, :] = new_input
    return Xin

first =1070
future=1153
forcast = []
Xin = Xtest[first:first+1,:,:]
for i in range(first,future+1):
    out = model.predict(Xin, batch_size=1)
    forcast.append(out[0,0])
    Xin = insert_end(Xin,out[0,0])
forcasted_output=np.asanyarray(forcast)
forcasted_output=forcasted_output.reshape(-1,1)
forcasted_output = scaler.inverse_transform(forcasted_output)

Ytest_1d = Ytest.flatten()
forcasted_output_1d = forcasted_output.flatten()

if choice == "Forecasting using Model":
    if st.checkbox("Complete Graph of Ytest and Forecast"):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(1, 1, figsize=(20, 9))
        ax.plot(range(0, 1093), Ytest_1d, 'b', label="History")
        ax.plot(range(1093, 1093 + 60), forcasted_output_1d[:60], 'r', label="Forecasted")
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        st.pyplot(fig)
    if st.checkbox("Forecast future 24 hours (1 Day)"):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(1, 1, figsize=(20, 9))
        ax.plot(range(1069, 1093), Ytest_1d[1069:1093], 'b', label="History", marker='o')
        ax.plot(range(1093, 1093 + 24), forcasted_output_1d[:24], 'r', label="Forecasted", marker='o')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        st.pyplot(fig)
    if st.checkbox("Forecast future 48 hours (2 Days)"):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(1, 1, figsize=(20, 9))
        ax.plot(range(1069, 1093), Ytest_1d[1069:1093], 'b', label="History", marker='o')
        ax.plot(range(1093, 1093 + 48), forcasted_output_1d[:48], 'r', label="Forecasted", marker='o')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        st.pyplot(fig)
    if st.checkbox("Temperature Values table of Forecast"):
        st.write(forcasted_output[0:72])

# REFERENCES
# https://www.tensorflow.org/tutorials/structured_data/time_series
# https://docs.streamlit.io/en/stable/
# https://www.tensorflow.org/guide/keras/save_and_serialize
# https://keras.io/examples/timeseries/timeseries_weather_forecasting/