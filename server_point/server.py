from flask import Flask,flash,redirect, jsonify, request
# from flask import Flask, jsonify
from flask_restx import Api, Resource
from flask_cors import CORS
import json
import numpy as np
from model import StockPrediction
import datetime
from finetune import fine_tune
from prediction import prediction 
from retrieve_data import retrieve_data
app = Flask(__name__)
CORS(app)  # Enable CORS for your app

api = Api(app, version='1.0', title='Your API', description='API Documentation')


nifty50 = ['ADANIPORTS', 'TITAN', 'ADANIENT', 'SBIN', 'BAJAJ-AUTO', 'DRREDDY', 'TATACONSUM', 'BAJAJFINSV', 'AXISBANK', 'HDFCLIFE', 'BRITANNIA', 'HINDALCO', 'CIPLA', 'TECHM', 'BHARTIARTL', 'TATASTEEL', 'HEROMOTOCO', 'BPCL', 'SUNPHARMA', 'M&M', 'HCLTECH', 'INDUSINDBK', 'COALINDIA', 'WIPRO', 'MARUTI', 'ASIANPAINT', 'ULTRACEMCO', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'HDFCBANK', 'NTPC', 'UPL', 'TATAMOTORS', 'BAJFINANCE', 'ONGC', 'JSWSTEEL', 'TCS', 'SBILIFE', 'GRASIM', 'EICHERMOT', 'KOTAKBANK', 'APOLLOHOSP', 'DIVISLAB', 'NESTLEIND', 'LT', 'RELIANCE', 'POWERGRID', 'LTIM', 'ITC']

niftyMidCap50 = ['RECLTD', 'BATAINDIA', 'ESCORTS', 'PFC', 'SHRIRAMFIN', 'PNB', 'BANDHANBNK', 'FEDERALBNK', 'MPHASIS', 'TRENT', 'M&MFIN', 'CONCOR', 'HINDPETRO', 'CUMMINSIND', 'IDFCFIRSTB', 'LTTS', 'LICHSGFIN', 'MRF', 'SAIL', 'GODREJPROP', 'ASHOKLEY', 'MFSL', 'UBL', 'ALKEM', 'TVSMOTOR', 'TATACOMM', 'GUJGASLTD', 'BIOCON', 'AUBANK', 'INDHOTEL', 'VOLTAS', 'NMDC', 'PETRONET', 'IDEA', 'OBEROIRLTY', 'HONAUT', 'ABCAPITAL', 'POLYCAB', 'BHARATFORG', 'AUROPHARMA', 'BALKRISIND', 'JUBLFOOD', 'ZEEL', 'ABBOTINDIA', 'PERSISTENT', 'OFSS', 'COFORGE', 'ZYDUSLIFE', 'ASTRAL', 'LUPIN']

niftySmallCap50 = ['LATENTVIEW', 'JUBLINGREA', 'MANAPPURAM', 'BSOFT', 'CHAMBLFERT', 'BSE', 'MRPL', 'KPITTECH', 'CAMS', 'AMARAJABAT', 'IEX', 'CYIENT', 'MAHABANK', 'PVRINOX', 'RBLBANK', 'IRB', 'CUB', 'IDFC', 'RVNL', 'ANGELONE', 'RENUKA', 'BIRLACORPN', 'GNFC', 'MEDANTA', 'ROUTE', 'ANURAS', 'UCOBANK', 'GRAPHITE', 'HINDCOPPER', 'METROPOLIS', 'RADICO', 'LXCHEM', 'CDSL', 'MAZDOCK', 'BIKAJI', 'JBCHEPHARM', 'CESC', 'BDL', 'HFCL', 'CAMPUS', 'MEDPLUS', 'UTIAMC', 'IDBI', 'NATIONALUM', 'WELSPUNIND', 'REDINGTON', 'SUZLON', 'BALAMINES', 'EASEMYTRIP', 'IOB']


app.config['SECRET_KEY']='0331a5c84e4e1924170f'


store_prediction = {}

@app.route("/store_predictions",methods = ["GET","POST"])
def store_predictions():
   store_prediction.clear()
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   fdays = 10
   pastdays = 100
   companyes = [nifty50,niftyMidCap50,niftySmallCap50]
   for groups in companyes:
      for cmp in groups:
         if cmp not in store_prediction:
                  future = json.loads(prediction(cmp,current_date,fdays))
                  past = json.loads(retrieve_data(company_name=cmp,date=current_date,days=pastdays))
                  store_prediction[cmp] = {"future":future,"past":past}
   return "done"
             
   
@app.route("/get_data_NIFTY_50_prediction",methods = ['GET','POST'])
def get_NIFTY_50_prediction():
   prediction_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   fdays = 10
   if request.args.get('fdays'):
      fdays = int(request.args.get('fdays'))
      
   pastdays = 0
   if request.args.get('pdays'):
      pastdays = int(request.args.get('pdays'))
   
   for cmp in nifty50:
      future = store_prediction[cmp]['future'][:fdays]
      if pastdays:
         past = store_prediction[cmp]['past'][:pastdays]
      prediction_data[cmp] = {"future":future,"past":past}
   json_data = jsonify(prediction_data) 
   return  json_data

@app.route("/get_NIFTY_50_data",methods = ['GET','POST'])
def get_NIFTY_50_data():
   days = int(request.args.get('days'))
   # companyname = request.args.get('company_name')
   retrieved_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   for cmp in nifty50:
      retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=days))
   json_data = jsonify(retrieved_data) 
   return  json_data

@app.route("/get_NIFTY_50_sigmoid",methods = ['GET','POST'])
def get_data_NIFTY_sigmoid():
   retrieved_data = {}
   days = 10
   prediction_data = {}
   profit = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   for cmp in nifty50:
      retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=1))
      prediction_data[cmp] = json.loads(prediction(cmp,current_date,days))
      profit[cmp] = int(prediction_data[cmp][-1]) - int(retrieved_data[cmp][0]['Close'])
      
   sorted_list_desc = sorted(profit.items(), key=lambda item: item[1], reverse=True)
   # print(sorted_list_desc)
   sorted_list_desc = sorted_list_desc[0:10]
   values = np.array([item[1] for item in sorted_list_desc])
   print(values)
   exp_values = np.exp(values-np.max(values))  # Subtracting the np.max value for numerical stability
   probabilities = exp_values / np.sum(exp_values)
   print(probabilities)
   result = [(sorted_list_desc[i][0], probabilities[i]) for i in range(len(sorted_list_desc))]
   final_return_val = {"prob":[],"comp":[],"cur_price":[]}
   for item in result:
      comp,prob = item
      cur_price =  retrieved_data[comp][0]['Close']
      final_return_val['prob'].append(prob)
      final_return_val['comp'].append(comp)
      final_return_val['cur_price'].append(cur_price)

   # print(f"cur : {retrieved_data['ADANIPORTS']}, prediction_data : {prediction_data['ADANIPORTS'][-1]} , {profit['ADANIPORTS']}")
   # print(f"curr : {retrieved_data['ADANIPORTS']}, {type(retrieved_data['ADANIPORTS'][0]['Close'])}, prediction_data : {prediction_data['ADANIPORTS'][-1]} , {type(prediction_data['ADANIPORTS'])}")
   json_data = jsonify(final_return_val) 
      # print(json_data)
   return json_data


@app.route("/get_small_cap_sigmoid",methods = ['GET','POST'])
def get_small_cap_sigmoid():
   small_retrieved_data = {}
   days = 10
   small_prediction_data = {}
   profit = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   for cmp in niftySmallCap50:
      small_retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=1))
      small_prediction_data[cmp] = json.loads(prediction(cmp,current_date,days))
      profit[cmp] = int(small_prediction_data[cmp][-1]) - int(small_retrieved_data[cmp][0]['Close'])
   
   sorted_list_desc = sorted(profit.items(), key=lambda item: item[1], reverse=True)
   # print(sorted_list_desc)
   sorted_list_desc = sorted_list_desc[0:10]
   values = np.array([item[1] for item in sorted_list_desc])
   print(values)
   exp_values = np.exp(values-np.max(values))  # Subtracting the np.max value for numerical stability
   probabilities = exp_values / np.sum(exp_values)
   print(probabilities)
   result = [(sorted_list_desc[i][0], probabilities[i]) for i in range(len(sorted_list_desc))]
   final_return_val = {"prob":[],"comp":[],"cur_price":[]}
   for item in result:
      comp,prob = item
      cur_price =  small_retrieved_data[comp][0]['Close']
      final_return_val['prob'].append(prob)
      final_return_val['comp'].append(comp)
      final_return_val['cur_price'].append(cur_price)

   json_data = jsonify(final_return_val) 
   return json_data

@app.route("/get_mid_cap_sigmoid",methods = ['GET','POST'])
def get_mid_cap_sigmoid():
   mid_retrieved_data = {}
   days = 10
   mid_prediction_data = {}
   profit = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   for cmp in niftyMidCap50:
      mid_retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=1))
      mid_prediction_data[cmp] = json.loads(prediction(cmp,current_date,days))
      profit[cmp] = int(mid_prediction_data[cmp][0]) - int(mid_retrieved_data[cmp][0]['Close'])
   
   sorted_list_desc = sorted(profit.items(), key=lambda item: item[1], reverse=True)
   # print(sorted_list_desc)
   sorted_list_desc = sorted_list_desc[0:10]
   values = np.array([item[1] for item in sorted_list_desc])
   exp_values = np.exp(values-np.max(values))  # Subtracting the np.max value for numerical stability
   probabilities = exp_values / np.sum(exp_values)
   result = [(sorted_list_desc[i][0], probabilities[i]) for i in range(len(sorted_list_desc))]
   final_return_val = {"prob":[],"comp":[],"cur_price":[]}
   for item in result:
      comp,prob = item
      cur_price =  mid_retrieved_data[comp][0]['Close']
      final_return_val['prob'].append(prob)
      final_return_val['comp'].append(comp)
      final_return_val['cur_price'].append(cur_price)

   json_data = jsonify(final_return_val) 
   return json_data

@app.route("/get_data_midcap_prediction",methods = ['GET','POST'])
def get_midcap_prediction():
   prediction_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   fdays = 10
   if request.args.get('fdays'):
      fdays = int(request.args.get('fdays'))
   pastdays = 0
   if request.args.get('pdays'):
      pastdays = int(request.args.get('pdays'))
   
   for cmp in niftyMidCap50:
      future = store_prediction[cmp]['future'][:fdays]
      if pastdays:
         past = store_prediction[cmp]['past'][:pastdays]
      prediction_data[cmp] = {"future":future,"past":past}
   json_data = jsonify(prediction_data) 
   return  json_data


@app.route("/get_midcap_data",methods = ['GET','POST'])
def get_midcap_data():
   days = int(request.args.get('days'))
   # companyname = request.args.get('company_name')
   retrieved_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   for cmp in niftyMidCap50:
      retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=days))
   json_data = jsonify(retrieved_data) 
   return  json_data

@app.route("/get_data_smallcap_prediction",methods = ['GET','POST'])
def get_smallcap_prediction():
   prediction_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   fdays = 10
   if request.args.get('fdays'):
      fdays = int(request.args.get('fdays'))
      
   pastdays = 0
   if request.args.get('pdays'):
      pastdays = int(request.args.get('pdays'))
   
   for cmp in niftySmallCap50:
      future = store_prediction[cmp]['future'][:fdays]
      if pastdays:
         past = store_prediction[cmp]['past'][:pastdays]
      prediction_data[cmp] = {"future":future,"past":past}
   json_data = jsonify(prediction_data) 
   return  json_data

@app.route("/get_smallcap_data",methods = ['GET','POST'])
def get_smallcap_data():
   days = int(request.args.get('days'))
   # companyname = request.args.get('company_name')
   retrieved_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   for cmp in niftySmallCap50:
      retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=days))
   json_data = jsonify(retrieved_data) 
   return  json_data

@app.route("/get_compant_prediction",methods = ['GET','POST'])
def get_data_company_prediction():
   prediction_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   # print(current_date)
   company_name = request.args.get("company_name")
   fdays = 10
   if request.args.get("fdays"):
      fdays =int(request.args.get("fdays"))
   pastdays = 0
   if request.args.get("pdays"):
      pastdays = int(request.args.get("pdays"))
   future = store_prediction[company_name]['future'][:fdays]
   if pastdays:
         past = store_prediction[company_name]['past'][:pastdays]
   prediction_data[company_name] = {"future":future,"past":past}
   json_data = jsonify(prediction_data) 
   return  json_data

   

@app.route("/get_current_data",methods = ['GET','POST'])
def get_current_data():
   days = int(request.args.get('days'))
   companyname = request.args.get('company_name')
   retrieved_data = {}
   current_date = datetime.datetime.now().strftime('%Y-%m-%d')
   # for cmp in nifty50:
   retrieved_data[companyname] = json.loads(retrieve_data(company_name=companyname,date=current_date,days=days))
   json_data = jsonify(retrieved_data) 
   return  json_data

@app.route("/finetune",methods=['GET'])
def finetune_models():
      # Get the current date
      current_date = datetime.datetime.now()
      # Calculate the previous day's date
      previous_day_date = current_date - datetime.timedelta(days=1)
      # Format the date as a string
      formatted_date = previous_day_date.strftime('%Y-%m-%d')
      args = request.args.get('company_name')
      if(args):
         fine_tune(date=formatted_date,company_name=args)
         return "done"
      
      companyes = [nifty50,niftyMidCap50,niftySmallCap50]
      for groups in companyes:
         for cmp in groups:
            fine_tune(date=formatted_date,company_name=cmp)
      return "done"

@api.route("/get_small_cap_sigmoid")
class small_cap_sigmoidresource(Resource):
   def get():
      small_retrieved_data = {}
      days = 10
      small_prediction_data = {}
      profit = {}
      current_date = datetime.datetime.now().strftime('%Y-%m-%d')
      for cmp in niftySmallCap50:
         small_retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=1))
         small_prediction_data[cmp] = json.loads(prediction(cmp,current_date,days))
         profit[cmp] = int(small_prediction_data[cmp][0]) - int(small_retrieved_data[cmp][0]['Close'])

      sorted_list_desc = sorted(profit.items(), key=lambda item: item[1], reverse=True)
      # print(sorted_list_desc)
      sorted_list_desc = sorted_list_desc[0:10]
      values = np.array([item[1] for item in sorted_list_desc])
      print(values)
      exp_values = np.exp(values-np.max(values)) # Subtracting the np.max value for numerical stability
      probabilities = exp_values / np.sum(exp_values)
      print(probabilities)
      result = [(sorted_list_desc[i][0], probabilities[i]) for i in range(len(sorted_list_desc))]
      final_return_val = {"prob":[],"comp":[],"cur_price":[]}
      for item in result:
         comp,prob = item
         cur_price =  small_retrieved_data[comp][0]['Close']
         final_return_val['prob'].append(prob)
         final_return_val['comp'].append(comp)
         final_return_val['cur_price'].append(cur_price)

      json_data = jsonify(final_return_val) 
      return json_data

@api.route("/get_mid_cap_sigmoid")
class mid_cap_sigmoidresource(Resource):
   def get():
      mid_retrieved_data = {}
      days = 10
      mid_prediction_data = {}
      profit = {}
      current_date = datetime.datetime.now().strftime('%Y-%m-%d')
      for cmp in niftyMidCap50:
         mid_retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=1))
         mid_prediction_data[cmp] = json.loads(prediction(cmp,current_date,days))
         profit[cmp] = int(mid_prediction_data[cmp][-1]) - int(mid_retrieved_data[cmp][0]['Close'])
   
      sorted_list_desc = sorted(profit.items(), key=lambda item: item[1], reverse=True)
      # print(sorted_list_desc)
      sorted_list_desc = sorted_list_desc[0:10]
      values = np.array([item[1] for item in sorted_list_desc])
      exp_values = np.exp(values-np.max(values)) # Subtracting the np.max value for numerical stability
      probabilities = exp_values / np.sum(exp_values)
      result = [(sorted_list_desc[i][0], probabilities[i]) for i in range(len(sorted_list_desc))]
      final_return_val = {"prob":[],"comp":[],"cur_price":[]}
      for item in result:
         comp,prob = item
         cur_price =  mid_retrieved_data[comp][0]['Close']
         final_return_val['prob'].append(prob)
         final_return_val['comp'].append(comp)
         final_return_val['cur_price'].append(cur_price)

      json_data = jsonify(final_return_val) 
      return json_data

@api.route('/finetune')
class finetuneResouce(Resource):
   def get():
      # Get the current date
      current_date = datetime.datetime.now()
      # Calculate the previous day's date
      previous_day_date = current_date - datetime.timedelta(days=1)
      # Format the date as a string
      formatted_date = previous_day_date.strftime('%Y-%m-%d')
      args = request.args.get('company_name')
      if(args):
         fine_tune(date=formatted_date,company_name=args)
         return "done"
      
      companyes = [nifty50,niftyMidCap50,niftySmallCap50]
      for groups in companyes:
         for cmp in groups:
            fine_tune(date=formatted_date,company_name=cmp)
      return "done"

@api.route('/get_NIFTY_50_prediction')
class NIFTY50PredictionResource(Resource):
    def get(self):
         prediction_data = {}
         current_date = datetime.datetime.now().strftime('%Y-%m-%d')
         days = 10
         for cmp in nifty50:
               prediction_data[cmp] = prediction(cmp, current_date, days)
         json_data = jsonify(prediction_data) 
         print(json_data)
         return json_data, 200


@api.route('/get_NIFTY_50_sigmoid')
class NIFTY50sigmoidResource(Resource):
    def put(self):
         
         retrieved_data = {}
         days = 10
         prediction_data = {}
         profit = {}
         current_date = datetime.datetime.now().strftime('%Y-%m-%d')
         for cmp in nifty50:
            retrieved_data[cmp] = json.loads(retrieve_data(company_name=cmp,date=current_date,days=1))
            prediction_data[cmp] = json.loads(prediction(cmp,current_date,days))
            profit[cmp] = int(prediction_data[cmp][-1]) - int(retrieved_data[cmp][0]['Close'])
      
         sorted_list_desc = sorted(profit.items(), key=lambda item: item[1], reverse=True)
         sorted_list_desc = sorted_list_desc[0:10]
         values = np.array([item[1] for item in sorted_list_desc])
         exp_values = np.exp(values-np.max(values)) # Subtracting the np.max value for numerical stability
         probabilities = exp_values / np.sum(exp_values)
         result = [(sorted_list_desc[i][0], probabilities[i]) for i in range(len(sorted_list_desc))]
         final_return_val = {"prob":[],"comp":[],"cur_price":[]}
         for item in result:
            comp,prob = item
            cur_price =  retrieved_data[comp][0]['Close']
            final_return_val['prob'].append(prob)
            final_return_val['comp'].append(comp)
            final_return_val['cur_price'].append(cur_price)

         # print(f"cur : {retrieved_data['ADANIPORTS']}, prediction_data : {prediction_data['ADANIPORTS'][-1]} , {profit['ADANIPORTS']}")
         # print(f"curr : {retrieved_data['ADANIPORTS']}, {type(retrieved_data['ADANIPORTS'][0]['Close'])}, prediction_data : {prediction_data['ADANIPORTS'][-1]} , {type(prediction_data['ADANIPORTS'])}")
         json_data = jsonify(final_return_val) 
            # print(json_data)
         return json_data, 200

@api.route('/get_data_midcap_prediction')
class midcapPredictionResource(Resource):
    def get(self):
         midcap_prediction_data = {}
         current_date = datetime.datetime.now().strftime('%Y-%m-%d')
         days = 10
         for cmp in niftyMidCap50:
            if count == 0:
               midcap_prediction_data[cmp] = prediction(cmp, current_date, days)
         json_data = jsonify(midcap_prediction_data)
         print(json_data)
         return json_data, 200
     

@api.route('/get_data_smallcap_prediction')
class smallcapPredictionResource(Resource):
    def get(self):
         smallcap_prediction_data = {}
         current_date = datetime.datetime.now().strftime('%Y-%m-%d')
         days = 10
         for cmp in niftySmallCap50:
            if count == 0:
               smallcap_prediction_data[cmp] = prediction(cmp, current_date, days)
           
         json_data = jsonify(smallcap_prediction_data)
         # print(json_data) 
         return json_data, 200
      
@api.route("/post_current_data")
class Companyretrive_cur_dataResource(Resource):
   def get(self):
      days = int(request.args.get('days'))
      # days = 7
      print(request.args.get('days'))
      companyname = request.args.get('company_name')
      print(companyname)
      retrieved_data = {}
      current_date = datetime.datetime.now().strftime('%Y-%m-%d')
      # for cmp in nifty50:
      retrieved_data[companyname] = retrieve_data(company_name=companyname,date=current_date,days=days)
      json_data = jsonify(retrieved_data) 
      return  json_data,200


@api.route('/get_company_prediction')
class CompanyPredictionResource(Resource):
    def get(self):
        Company_prediction_data = {}
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        company_name = api.payload.get("company_name")
        days = api.payload.get("days")
        Company_prediction_data[current_date] = prediction(company_name, current_date, days)
        print(Company_prediction_data)
        json_data_company = jsonify(Company_prediction_data) 
        return json_data_company, 200



if __name__ == '__main__':
    app.run(debug=True, port=8000)
