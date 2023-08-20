import os
import json
from flask import Flask,flash,redirect, jsonify, request
import yfinance as yf
from datetime import datetime, timedelta
from prediction import prediction
# from getCurrentPrice import get_current_price
import pandas as pd
import numpy as np


def retrieve_data(date, company_name, days=100, future=False):
    if future:
        predicted_data = prediction(date, company_name, days)
        return predicted_data
    else:
        company_name_extension = company_name + ".NS"
        input_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = input_date - timedelta(days=150)
        start_date = start_date.strftime("%Y-%m-%d")
        input_date = input_date.strftime("%Y-%m-%d")
        prev_data = yf.download(company_name_extension, start=start_date, end=input_date)
        
        if days > 100:
            prev_data = prev_data.tail(100)
        else:
            prev_data = prev_data.tail(days)
            prev_data = prev_data.reset_index()
            prev_data = prev_data[['Date', 'Close']]
            prev_data['Date'] = pd.to_datetime(prev_data['Date']).astype(str)
            prev_data['Close'] = prev_data['Close'].astype(float)
        data_dict = prev_data.to_dict(orient='records')
        # print( f"data_dict {data_dict} , {type(data_dict)}")
        # print(f"res : {res} , {type(res)}" )
        return json.dumps(data_dict)

    
if __name__== "__main__":
    cur_date = datetime.now().strftime('%Y-%m-%d')
    company = "AXISBANK"
    a = retrieve_data(cur_date,company,10)
    # print(a)