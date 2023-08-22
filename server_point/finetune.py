import os
import torch
import torch.nn as nn
import pickle
from model import StockPrediction
import yfinance as yf
import json
from datetime import datetime, timedelta
import sys
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_model(directory, target_filename):
    file = f'{target_filename}.NS_data.h5'
    pickle_file = os.path.join(directory, file)
    # print(f"pickle_file {pickle_file}")
    
    if os.path.exists(pickle_file):
        # print("valid path")
        # checkpoint = 
        return torch.load(pickle_file)
    
    else:
        return None
    

def store_model(model,directory,target_filename):
    pickle_file = os.path.join(directory, f'{target_filename}.NS_data.h5')
    torch.save(model,pickle_file )
    
def prepare_dataframe_for_prediction(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace = True)
    return df


def fine_tune(date, company_name, days=60):
    learning_rate = 0.001
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

    prev_data['Close'] = (prev_data['Close']-mean)/st_deviation
    shifted_df = prepare_dataframe_for_prediction(prev_data, 60)
    shifted_df_as_np = shifted_df.to_numpy()
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    
    pickle_directory = "../path_to_store_pickle_files/"
    pkl_file_name = f'{company_name}'
    model_ = load_model(pickle_directory, pkl_file_name)
    # print(model_)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model_.parameters(), lr=learning_rate)
    model_.train()
    # print(X.shape)
    # print(X)
    X = X.unsqueeze(dim=2)
    # print(f"X.shape : {X.shape}")/
    # X = X.transpose(0, 2)
    # print(f"X.shape : {X.shape}")
    y = y.unsqueeze(dim=0)
    # print(f"y.shape : {y.shape}")
    # y = y.transpose(0, 1)
    # print(f"y.shape : {y.shape}")
    # print(X)
    # print(y)
    # print(X.shape)
    # print(X.dtype)
    # print(y.shape)
    output = model_(X)
    print(f"predicted : {output} , true : {y}")
    loss = loss_function(output, y)
    # running_loss += loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    # store_model(model_,pickle_directory,pkl_file_name)

if __name__  == "__main__":
    
    company = "AXISBANK"
    fine_tune("2023-08-12", company, 10)