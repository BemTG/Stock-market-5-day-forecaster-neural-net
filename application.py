from flask import Flask, render_template, flash, request, url_for, Markup
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging, io, base64, os, datetime
from datetime import datetime
from datetime import timedelta
from lxml import html
import time, os, string, requests
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

# plt.style.use('ggplot')



# Initialize the Flask application
app = application= Flask(__name__)

# global variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROLLING_PERIOD = 10
PREDICT_OUT_PERIOD = 5
SAVED_REGRESSION_MODEL_PATH = 'model.json'
SAVED_MODEL_WEIGHTS_PATH = "model.h5"
FEATURES = [str(id) for id in range(0,ROLLING_PERIOD)]

model_features = None
stock_market_historical_data = None
stock_market_live_data = None
options_stocks = None
predict_fn = None

# load nasdaq corollary material
stock_company_info_amex = None
stock_company_info_nasdaq = None
stock_company_info_nyse = None

def load_fundamental_company_info():
    global stock_company_info_amex, stock_company_info_nasdaq, stock_company_info_nyse
    import pandas as pd
    stock_company_info_amex = pd.read_csv(os.path.join(BASE_DIR, 'stock_company_info_amex.csv'))
    stock_company_info_nasdaq = pd.read_csv(os.path.join(BASE_DIR, 'stock_company_info_nasdaq.csv'))
    stock_company_info_nyse = pd.read_csv(os.path.join(BASE_DIR, 'stock_company_info_nyse.csv'))

def get_fundamental_information(symbol):
    CompanyName = "No company name"
    Sector = "No sector"
    Industry = "No industry"
    MarketCap = "No market cap"
    Exchange = 'No exchange'

    if (symbol in list(stock_company_info_nasdaq['Symbol'])):
        data_row = stock_company_info_nasdaq[stock_company_info_nasdaq['Symbol'] == symbol]
        CompanyName = data_row['Name'].values[0]
        Sector = data_row['Sector'].values[0]
        Industry = data_row['Industry'].values[0]
        MarketCap = data_row['MarketCap'].values[0]
        Exchange = 'NASDAQ'

    elif (symbol in list(stock_company_info_amex['Symbol'])):
        data_row = stock_company_info_amex[stock_company_info_amex['Symbol'] == symbol]
        CompanyName = data_row['Name'].values[0]
        Sector = data_row['Sector'].values[0]
        Industry = data_row['Industry'].values[0]
        MarketCap = data_row['MarketCap'].values[0]
        Exchange = 'AMEX'

    elif (symbol in list(stock_company_info_nyse['Symbol'])):
        data_row = stock_company_info_nyse[stock_company_info_nyse['Symbol'] == symbol]
        CompanyName = data_row['Name'].values[0]
        Sector = data_row['Sector'].values[0]
        Industry = data_row['Industry'].values[0]
        MarketCap = data_row['MarketCap'].values[0]
        Exchange = 'NYSE'

    return (CompanyName, Sector, Industry, MarketCap, Exchange)

# !sudo pip3 install wikipedia
def get_wikipedia_intro(symbol):
    import wikipedia
    company_name = get_fundamental_information(symbol)[0]
    if (company_name is not None):
            description = wikipedia.page(company_name).content
    else:
        return (str('No Data Available'))

    return(description.split('\n')[0])

def get_stock_prediction(symbol):
    temp_df=GetLiveStockData(symbol, size=50)
    #temp_df = stock_market_live_data[stock_market_live_data['symbol']==symbol]

    # forecast on live data (basically the last x days where we don't have an outcome...)
    predictions = predict_fn.predict(temp_df[model_features])
    forecasts =  [ item for elem in predictions.tolist() for item in elem]

    return(forecasts)

def GetLiveStockData(symbol, size=50):
    # we'll use pandas_datareader
    import datetime
    pd.core.common.is_list_like = pd.api.types.is_list_like
    import pandas_datareader.data as web

    try:
        end = datetime.datetime.now()
        start = datetime.datetime.now() - datetime.timedelta(days=60)
        live_stock_data = web.DataReader(symbol, 'yahoo', start, end)
        live_stock_data.reset_index(inplace=True)
        live_stock_data = live_stock_data[[ 'Date', 'Adj Close']]
        live_stock_data=live_stock_data.dropna()
        # live_stock_data = live_stock_data[['symbol', 'begins_at', 'close_price']]
        # live_stock_data.columns = ['symbol', 'date', 'close']
    except:
        live_stock_data = None

    if (live_stock_data is not None):
        live_stock_data = live_stock_data.sort_values('Date')
        live_stock_data = live_stock_data.tail(size)

        # make data model ready
        #live_stock_data['Close'] = pd.to_numeric(live_stock_data['Close'], errors='coerce')
        live_stock_data['Adj Close'] = np.log(live_stock_data['Adj Close'])

        # clean up the data so it aligns with our earlier notation
        #live_stock_data['date'] = pd.to_datetime(live_stock_data['date'], format = '%m/%d/%y')
        # sort by ascending dates as we've done in training
        live_stock_data = live_stock_data.sort_values('Date')

        # build dataset
        X = []

        prediction_dates = []
        last_market_dates = []

        # rolling predictions
        rolling_period = 10
        predict_out_period = 5

        for per in range(rolling_period, len(live_stock_data)):
            X_tmp = []
            for rollper in range(per-rolling_period,per):
                # build the 'features'
                X_tmp += [live_stock_data['Adj Close'].values[rollper]]

            X.append(np.array(X_tmp))

            # add x days to last market date using numpy timedelta64
            prediction_dates.append(live_stock_data['Date'].values[per] + np.timedelta64(predict_out_period,'D'))
            last_market_dates.append(live_stock_data['Date'].values[per])


        live_stock_ready_df = pd.DataFrame(X)
        live_stock_ready_df.columns = [str(f) for f in list(live_stock_ready_df)]

        live_stock_ready_df['prediction_date'] = prediction_dates
        live_stock_ready_df['last_market_date'] = last_market_dates

        # live_stock_ready_df.columns = [str(f) for f in live_stock_ready_df]

        return(live_stock_ready_df)
    else:
        return(None)

def get_plot_prediction(symbol):

    predictions = get_stock_prediction(symbol)

    if (len(predictions) > 0):
        # temp_df = stock_market_live_data[stock_market_live_data['symbol']==symbol]
        temp_df=GetLiveStockData(symbol, size=50)
            
        actuals = list(temp_df.tail(1).values[0])[0:ROLLING_PERIOD]
        # actuals = list(temp_df[FEATURES].values[0])
        # transform log price to price of past data
        actuals = list(np.exp(actuals))

        days_before = temp_df['last_market_date'].values[-1]
        days_before_list = []
        for d in range(ROLLING_PERIOD):
            days_before_list.append(str(np.busday_offset(np.datetime64(days_before,'D'),-d, roll='backward')))
            days_before_list.sort()

        fig, ax = plt.subplots(figsize=(9,3))
        plt.plot(days_before_list, actuals, color='teal', linewidth=2.5)

        for d in range(1, PREDICT_OUT_PERIOD+1):
            days_before_list.append(str(np.busday_offset(np.datetime64(days_before,'D'),d, roll='forward')))
            actuals.append(np.exp(predictions[-1]))

        plt.suptitle('Forecast for ' + str(temp_df['prediction_date'].values[-1])[0:10] + '     $' +
                     str(np.round(np.exp(predictions[-1]),2)))

        ax.plot(days_before_list, actuals, color='teal', linestyle='dashed')
        ax.grid()

        plt.xticks(days_before_list, days_before_list, fontsize = 7)
        # ax.set_xticklabels(days_before_list, rotation = 35, ha="right")
        fig.autofmt_xdate()


        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_bit_to_text = base64.b64encode(img.getvalue()).decode()

        chart_plot = Markup('<img style="padding:0px; width: 80%; height: 500px" src="data:image/png;base64,{}">'.format(plot_bit_to_text))
        return(chart_plot)

@app.before_first_request
def prepare_data():
    global stock_market_historical_data, stock_market_live_data, options_stocks, predict_fn, model_features
    stock_market_historical_data = pd.read_csv(os.path.join(BASE_DIR, 'stock_market_historical_data.csv'))
    # stock_market_live_data = pd.read_csv(os.path.join(BASE_DIR, 'stock_market_live_data.csv'))
    # stock_market_live_data=GetLiveStockData(symbol, size=50)

    options_stocks = sorted(set(stock_market_historical_data['symbol']))
    # option_stocks=pd.read_csv('sp500.csv')
    # option_stocks=option_stocks['Stocks']
    # option_stocks=option_stocks.tolist()
    load_fundamental_company_info()

    # deserialize the model
    model_features = [f for f in list(stock_market_historical_data) if f not in ['outcome', 'last_market_date', 'symbol', 'prediction_date' ]]
    json_file = open(SAVED_REGRESSION_MODEL_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    predict_fn = model_from_json(loaded_model_json)
    # load weights into new model
    predict_fn.load_weights(SAVED_MODEL_WEIGHTS_PATH)

@application.route('/', methods=['POST','GET'])
def GetForecastHome():


    return render_template('intro.html')
    
    
@application.route('/error', methods=['POST','GET'])
def GetForecastError():


    return render_template('error.html')
    


# Define a route for the default URL, which loads the form
@app.route('/forecast', methods=['POST', 'GET'])
def get_financial_information():
    chart_plot = ''
    fundamentals_company_name = ''
    fundamentals_sector = ''
    fundamentals_industry = ''
    fundamentals_marketcap = ''
    fundamentals_exchange = ''

    wiki = ''
    selected_stock = ''

    if request.method == 'POST':
        selected_stock = request.form['selected_stock']
        fundamentals = get_fundamental_information(selected_stock)

        if not  fundamentals[0] is np.NaN:
            fundamentals_company_name = Markup("<b>" + str(fundamentals[0]) + "</b><BR><BR>")
        if not fundamentals[1] is np.NaN:
            fundamentals_sector = Markup("Sector: <b>" + str(fundamentals[1]) + "</b><BR><BR>")
        if not fundamentals[2] is np.NaN:
            fundamentals_industry = Markup("Industry: <b>" + str(fundamentals[2]) + "</b><BR><BR>")
        if not fundamentals[3] is np.NaN:
            fundamentals_marketcap = Markup("MarketCap: <b>$" + str(fundamentals[3]) + "</b><BR><BR>")
        if not fundamentals[4] is np.NaN:
            fundamentals_exchange = Markup("Exchange: <b>" + str(fundamentals[4]) + "</b><BR><BR>")
        
        chart_plot = get_plot_prediction(selected_stock)
        wiki = get_wikipedia_intro(selected_stock)
        if wiki==None:
            return render_template('error.html')


    return render_template('stock-market-report.html',
        options_stocks=options_stocks,
        selected_stock = selected_stock,
        chart_plot=chart_plot,
        wiki = wiki,
        fundamentals_company_name = fundamentals_company_name,
        fundamentals_sector = fundamentals_sector,
        fundamentals_industry = fundamentals_industry,
        fundamentals_marketcap = fundamentals_marketcap,
        fundamentals_exchange = fundamentals_exchange)
        # elif:
        #     return render_template('error.html')




    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)