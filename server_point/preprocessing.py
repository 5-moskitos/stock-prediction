from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pickle
import os





def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace = True)
    return df


def preprocessing(csv_file_path, lookback):
    parameters = {}
    parameters_directory = "../parameters"

    if not os.path.exists(parameters_directory+'/parameters.pkl'):
        os.makedirs(parameters_directory, exist_ok=True)

    try:
        with open(os.path.join(parameters_directory, 'parameters.pkl'), 'rb') as f:
            unpickler = pickle.Unpickler(f)
            parameters = unpickler.load()
    except EOFError:
        pass
    


    filename = csv_file_path.split('.csv')[0]
    normalizaiton_parameters = []


    df = pd.read_csv(csv_file_path)
    df = df[['Date', 'Close']]
    normalizaiton_parameters.append(df.shape[0])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close'].astype(float)
    
    mean = df['Close'].mean()
    st_deviation = df['Close'].std()
    df['Close'] = (df['Close'] - mean)/st_deviation
    shifted_df = prepare_dataframe_for_lstm(df, lookback)

    shifted_df_as_np = shifted_df.to_numpy()


    normalizaiton_parameters.append(mean)
    normalizaiton_parameters.append(st_deviation)
    filename = filename.split('.NS')[0].strip()
    if '/' in filename:
        filename = filename.split('/')[1]
    parameters[filename] = normalizaiton_parameters

    with open(os.path.join(parameters_directory, 'parameters.pkl'), 'wb') as f:
        pickle.dump(parameters, f)

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    return X, y


if __name__ == "__main__":
    
    for csv_file in os.listdir('.'):
        if csv_file.endswith('.csv'):
            csv_file_path = os.path.join('.', csv_file)
            X, y = preprocessing(csv_file_path, 60)