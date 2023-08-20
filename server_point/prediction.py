import os
import torch
import pickle
from model import StockPrediction
import yfinance as yf
import json
from datetime import datetime, timedelta
import sys
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



def load_pickle_file(directory, target_filename):
    pickle_file = os.path.join(directory, f'{target_filename}.NS_data.h5')
    # print(f"pickle_file {pickle_file}")
    
    if os.path.exists(pickle_file):
        # print("valid path")
        checkpoint = torch.load(pickle_file)
        return checkpoint
    
    else:
        return None


def prepare_dataframe_for_prediction(prev_data, n_steps):
    prev_data = dc(prev_data)

    prev_data.set_index('Date', inplace=True)
    for i in range(1, n_steps+1):
        prev_data[f'Close(t-{i})'] = prev_data['Close'].shift(i)


    prev_data.dropna(inplace = True)
    return prev_data





def prediction(company_name, date,  days=60):

    company_name_extension = company_name + ".NS"
    input_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = input_date - timedelta(days=365)
    start_date = start_date.strftime("%Y-%m-%d")
    input_date = input_date.strftime("%Y-%m-%d")
    prev_data = yf.download(company_name_extension, start=start_date, end=input_date)

    prev_data = prev_data.tail(61)
    prev_data = prev_data.reset_index()
    prev_data = prev_data[['Date', 'Close']]
    prev_data['Date'] = pd.to_datetime(prev_data['Date'])
    prev_data['Close'] = prev_data['Close'].astype(float)
    parameter_file = '../parameters'
    pickle_filename = pickle_file = os.path.join(parameter_file, 'parameters.pkl')
    
    with open(pickle_filename, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)

    parameters_list = loaded_dict[company_name]
    size = parameters_list[0]
    mean = parameters_list[1]
    st_deviation = parameters_list[2]

    # print(prev_data)
    prev_data['Close'] = (prev_data['Close']-mean)/st_deviation

    shifted_df = prepare_dataframe_for_prediction(prev_data, 60)
    shifted_df_as_np = shifted_df.to_numpy()

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    pickle_directory = "../path_to_store_pickle_files/"
    pkl_file_name = f'{company_name}'
    model_ = load_pickle_file(pickle_directory, pkl_file_name)
    X = X.unsqueeze(-1)

    predicted_data = []

    if days > 60:
        days = 60

    for i in range(days):
        output = model_(X)

        shifted_X = X[:, 1:, :]
        X = torch.cat((shifted_X, output.unsqueeze(0)), dim=1)
        output = (output*st_deviation) + mean
        predicted_data.append(output.item())
        
    return json.dumps(predicted_data)
    




if __name__  == "__main__":
    company = "AXISBANK"
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f" {type(current_date)}, {current_date}")
    # predicted_data = prediction("2023-08-18", company, 10)
    # print(predicted_data)
