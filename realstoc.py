
from pyexpat import model
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
import datetime
import cufflinks as cf
from streamlit_option_menu import option_menu






with st.sidebar:
    choose = option_menu ("Navigation", ["About", "Stock Analysis", "Prediction Trend Graph","Forecasting","Feedback"],
                         icons=['cursor-fill','bar-chart', 'kanban','arrow-right-square-fill','person-rolodex'],
                         menu_icon="app-indicator", default_index=0,
                         
                         styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "": "#eee"},
        "nav-link-selected": {"background-color": "#00A4CCFF"},
    }
    )
logo = Image.open(r'E:\shortcuts\stockpriceproject\Sourcecode\imh.png')

if choose == "About":
    st.markdown(""" <style> .title {
    font-size:35px ; font-family: 'Bradley Hand ITC'; color: yellow;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<h1 class="title"> WELCOME TO STOCKERS</h1>',unsafe_allow_html=True)
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               
      # To display the header text using css style
        st.markdown(""" <style> .font {
                      font-size:35px ; font-family: 'TimesNewRoman'; color: #FF9633;} 
                        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About Our project</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
      st.image(logo, width=130 )
      st.subheader("What is stock ?")
      st.write("A stock (also known as equity) is a security that represents the ownership of a fraction of a corporation. This entitles the owner of the stock to a proportion of the corporation's assets and profits equal to how much stock they own. Units of stock are called 'shares' ")
      st.subheader("Stock Prediction ") 
      st.write("Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stock's future price could yield significant profit. The efficient-market hypothesis suggests that stock prices reflect all currently available information and any price changes that are not based on newly revealed information thus are inherently unpredictable. Others disagree and those with this viewpoint possess myriad methods and technologies which purportedly allow them to gain future price information.")
      st.image("https://images.news18.com/ibnlive/uploads/2021/08/market-2-16298997324x3.jpg")

if choose== "Prediction Trend Graph":
    st.markdown(""" <style> .title {
    font-size:35px ; font-family: 'Bradley Hand ITC'; color: yellow;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<h1 class="title"> STOCKERS</h1>',unsafe_allow_html=True)
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
      st.markdown(""" <style> .font {
                      font-size:35px ; font-family: 'TimesNewRoman'; color: #FF9633;} 
                        </style> """, unsafe_allow_html=True)
      st.markdown('<p class="font">TREND PREDICTION GRAPH</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
       st.image(logo, width=130 )


    start ='2020-01-01'
    end='2023-12-31'
  
    user_input=st.text_input('Enter Stock Ticker')
    df=data.DataReader(user_input,'yahoo',start,end)

    name = yf.Ticker(user_input)

    com_name = name.info['longName']
    st.subheader('Selected Company Name is '+"    "+com_name)



    st.subheader('Data from 2010-2023')
    st.write(df.describe())
    st.subheader('closing price vs time chart')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('closing price vs timechart100MA')
    ma100=df.Close.rolling(100).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('closing price vs timechart100MA&200MA')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)
    st.write(ma100)

    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    data_training_array=scaler.fit_transform(data_training)

    x_train=[]
    y_train=[]


    for i in range(100,data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])
    x_train,y_train=np.array(x_train),np.array(y_train)    

    model=load_model('keras_model.h5')

    past_100_days=data_training.tail(100)
    final_df=past_100_days.append(data_testing,ignore_index=True)
    input_data=scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]

    for i in range(100,input_data.shape[0]):
         x_test.append(input_data[i-100:i])
         y_test.append(input_data[i,0])

    x_test,y_test=np.array(x_test),np.array(y_test)
    y_predicted=model.predict(x_test)
    scaler=scaler.scale_

    scale_factor=1/scaler[0]
    y_predicted=y_predicted*scale_factor
    y_test=y_test*scale_factor

    st.subheader('Predictions')

    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig2)

if choose=="Stock Analysis":
        st.markdown(""" <style> .title {
        font-size:35px ; font-family: 'Bradley Hand ITC'; color: yellow;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<h1 class="title"> STOCKERS</h1>',unsafe_allow_html=True)
        col1, col2 = st.columns( [0.8, 0.2])
        with col1: 
                        # To display the header text using css style
         st.markdown(""" <style> .font {
                      font-size:35px ; font-family: 'TimesNewRoman'; color: #FF9633;} 
                        </style> """, unsafe_allow_html=True)
         st.markdown('<p class="font">About stocks upto to today</p>', unsafe_allow_html=True)    
        with col2:               # To display brand log
          st.image(logo, width=130 )


        
        st.sidebar.subheader('Query parameters')
        start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
        end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
        ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
        tickerSymbol = st.selectbox('Stock ticker', ticker_list) # Select ticker symbol
        tickerData = yf.Ticker(tickerSymbol) # Get ticker data
        tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# Ticker information
        string_logo = '<img src=%s>' % tickerData.info['logo_url']
        st.markdown(string_logo, unsafe_allow_html=True)

        string_name = tickerData.info['longName']
        st.header('**%s**' % string_name)

        string_summary = tickerData.info['longBusinessSummary']
        st.info(string_summary)

# Ticker data
        st.header('**Ticker data**')
        st.write(tickerDf)

# Bollinger bands
        st.header('**Bollinger Bands**')
        qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)


if choose == "Feedback":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'TimesNewRoman'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> .title {
    font-size:35px ; font-family: 'Bradley Hand ITC'; color: yellow;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<h1 class="title"> STOCKERS</h1>',unsafe_allow_html=True)
    st.markdown('<p class="font">Feedback Form</p>', unsafe_allow_html=True)
 
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
       
        Message=st.text_input(label='Please Enter Your Feedback') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your Feedback')
if choose == "Forecasting":
  import streamlit as st
  import time
  from tqdm.notebook import tqdm
  from tensorflow import keras
  import datetime as dt
  from datetime import date
  import yfinance as yf
  import pandas as pd
  from plotly import graph_objs as go
  import  plotly.express as px
  import math
  import numpy as np
  from sklearn.preprocessing import MinMaxScaler
  from keras.models import Sequential
  from keras.layers import Dense, LSTM
  import matplotlib.pyplot as plt


  START = "2020-01-01"
#PREVIOUS=
  TODAY = dt.datetime.now().strftime("%Y-%m-%d")
  NEXTDAY =datetime. datetime. today() + datetime. timedelta(days=1)
  PREVIOUSDAY=datetime. datetime. today() + datetime. timedelta(days=1)
  NXT=NEXTDAY.date()
  PRE=PREVIOUSDAY.date()

  st.title("Stock Prediction App")

  stocks = ["Select the Stock", "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "GME", "NVDA", "AMD"]


# Loading Data ---------------------

#@st.cache(suppress_st_warning=True)
  def load_data(ticker):
    data = yf.download(ticker, START,  TODAY)
    data.reset_index(inplace=True)
    return data


#For Stock Financials ----------------------

  def stock_financials(stock):
    df_ticker = yf.Ticker(stock)
    sector = df_ticker.info['sector']
    prevClose = df_ticker.info['previousClose']
    marketCap = df_ticker.info['marketCap']
    twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
    fiftyTwoWeekHigh = df_ticker.info['fiftyTwoWeekHigh']
    fiftyTwoWeekLow = df_ticker.info['fiftyTwoWeekLow']
    Name = df_ticker.info['longName']
    averageVolume = df_ticker.info['averageVolume']
    ftWeekChange = df_ticker.info['52WeekChange']
    website = df_ticker.info['website']

    st.write('Company Name -', Name)
    st.write('Sector -', sector)
    st.write('Company Website -', website)
    st.write('Average Volume -', averageVolume)
    st.write('Market Cap -', marketCap)
    st.write('Previous Close -', prevClose)
    st.write('52 Week Change -', ftWeekChange)
    st.write('52 Week High -', fiftyTwoWeekHigh)
    st.write('52 Week Low -', fiftyTwoWeekLow)
    st.write('200 Day Average -', twoHunDayAvg)


#Plotting Raw Data ---------------------------------------

  def plot_raw_data(stock, data_1):
    df_ticker = yf.Ticker(stock)
    Name = df_ticker.info['longName']
    #data1 = df_ticker.history()
    data_1.reset_index()
    #st.write(data_1)
    numeric_df = data_1.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns.tolist()
    st.markdown('')
    st.markdown('**_Features_** you want to **_Plot_**')
    features_selected = st.multiselect("", numeric_cols)
    if st.button("Generate Plot"):
        cust_data = data_1[features_selected]
        plotly_figure = px.line(data_frame=cust_data, x=data_1['Date'], y=features_selected,
                                title= Name + ' ' + '<i>timeline</i>')
        plotly_figure.update_layout(title = {'y':0.9,'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
        plotly_figure.update_xaxes(title_text='Date')
        plotly_figure.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, title="Price"), width=800, height=550)
        st.plotly_chart(plotly_figure)


#For LSTM MOdel ------------------------------

  def create_train_test_LSTM(df, epoch, b_s, ticker_name):

    df_filtered = df.filter(['Close'])
    dataset = df_filtered.values

    #Training Data
    training_data_len = math.ceil(len(dataset) * .7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(train_data[i-60:i, 0])
        y_train_data.append(train_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    #Testing Data
    test_data = scaled_data[training_data_len - 60:, :]

    x_test_data = []
    y_test_data = dataset[training_data_len:, :]

    for j in range(60, len(test_data)):
        x_test_data.append(test_data[j - 60:j, 0])

    x_test_data = np.array(x_test_data)

    x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train_data, y_train_data, batch_size=int(b_s), epochs=int(epoch))
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted vs Actual Results for LSTM")
    st.write("Stock Prediction on Test Data for - ",ticker_name)

    predictions = model.predict(x_test_data)
    predictions = scaler.inverse_transform(predictions)

    train = df_filtered[:training_data_len]
    valid = df_filtered[training_data_len:]
    valid['Predictions'] = predictions

    new_valid = valid.reset_index()
    new_valid.drop('index', inplace=True, axis=1)
    st.dataframe(new_valid)
    st.markdown('')
    st.write("Plotting Actual vs Predicted ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(14, 8))
    plt.title('Actual Close prices vs Predicted Using LSTM Model', fontsize=20)
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size":20})
    st.pyplot()



#Creating Training and Testing Data for other Models ----------------------
 
  def create_train_test_data(df1):

    data = df1.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df1)), columns=['Date', 'High', 'Low', 'Open', 'Volume', 'Close'])

    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['High'][i] = data['High'][i]
        new_data['Low'][i] = data['Low'][i]
        new_data['Open'][i] = data['Open'][i]
        new_data['Volume'][i] = data['Volume'][i]
        new_data['Close'][i] = data['Close'][i]

    #Removing the hour, minute and second
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date

    train_data_len = math.ceil(len(new_data) * .8)

    train_data = new_data[:train_data_len]
    test_data = new_data[train_data_len:]

    return train_data, test_data


#Finding Movinf Average ---------------------------------------

  def find_moving_avg(ma_button, df):
    days = ma_button

    data1 = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data1)):
        new_data['Date'][i] = data1['Date'][i]
        new_data['Close'][i] = data1['Close'][i]

    new_data['SMA_'+str(days)] = new_data['Close'].rolling(min_periods=1, window=days).mean()

    #new_data.dropna(inplace=True)
    new_data.isna().sum()

    #st.write(new_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=new_data['Date'], y=new_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=new_data['Date'], y=new_data['SMA_'+str(days)], mode='lines', name='SMA_'+str(days)))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)


#Finding Linear Regression ----------------------------

  def Linear_Regression_model(train_data, test_data):

    x_train = train_data.drop(columns=['Date', 'Close'], axis=1)
    x_test = test_data.drop(columns=['Date', 'Close'], axis=1)
    y_train = train_data['Close']
    y_test = test_data['Close']

    #First Create the LinearRegression object and then fit it into the model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train, y_train)

    #Making the Predictions
    prediction = model.predict(x_test)

    return prediction


#Plotting the Predictions -------------------------


  def prediction_plot(pred_data, test_data, models, ticker_name):

    test_data['Predicted'] = 0
    test_data['Predicted'] = pred_data

    #Resetting the index
    test_data.reset_index(inplace=True, drop=True)
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted Price vs Actual Close Price Results for - " ,models)
    st.write("Stock Prediction on Test Data for - ", ticker_name)
    st.write(test_data[['Date', 'Close', 'Predicted']])
    st.write("Plotting Close Price vs Predicted Price for - ", models)

    #Plotting the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)



# Sidebar Menu -----------------------

  menu=["Stock Exploration and Feature Extraction", "Train Model"]
  st.sidebar.title("Settings")
  st.sidebar.subheader("Timeseries Settings")
  choices = st.sidebar.selectbox("Select the Activity", menu,index=0)



  if choices == 'Stock Exploration and Feature Extraction':
    st.subheader("Extract Data")
    #user_input = ''
    st.markdown('Enter **_Ticker_ Symbol** for the **Stock**')
    #user_input=st.selectbox("", stocks)
    user_input = st.text_input("", '')

    if not user_input:
        pass
    else:
        data = load_data(user_input)
        st.markdown('Select from the options below to Explore Stocks')

        selected_explore = st.selectbox("", options=['Select your Option', 'Stock Financials Exploration',
                                                     'Extract Features for Stock Price Forecasting'], index=0)
        if selected_explore == 'Stock Financials Exploration':
            st.markdown('')
            st.markdown('**_Stock_ _Financial_** Information ------')
            st.markdown('')
            st.markdown('')
            stock_financials(user_input)
            plot_raw_data(user_input, data)
            st.markdown('')
            shw_SMA = st.checkbox('Show Moving Average')

            if shw_SMA:
                st.write('Stock Data based on Moving Average')
                st.write('A Moving Average(MA) is a stock indicator that is commonly used in technical analysis')
                st.write(
                    'The reason for calculating moving average of a stock is to help smooth out the price of data over '
                    'a specified period of time by creating a constanly updated average price')
                st.write(
                    'A Simple Moving Average (SMA) is a calculation that takes the arithmatic mean of a given set of '
                    'prices over the specified number of days in the past, for example: over the previous 15, 30, 50, '
                    '100, or 200 days.')

                ma_button = st.number_input("Select Number of Days Moving Average", 5, 200)

                if ma_button:
                    st.write('You entered the Moving Average for ', ma_button, 'days')
                    find_moving_avg(ma_button, data)

        elif selected_explore == 'Extract Features for Stock Price Forecasting':
            st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
            start_date = st.date_input("", date(2022, 6,14))
            st.write('You Selected Data From - ', start_date)
            submit_button = st.button("Extract Features")

            start_row = 0
            if submit_button:
                st.write('Extracted Features Dataframe for ', user_input)
                for i in range(0, len(data)):
                    if start_date <= pd.to_datetime(data['Date'][i]):
                        start_row = i
                        break
                # data = data.set_index(pd.DatetimeIndex(data['Date'].values))
                st.write(data.iloc[start_row:, :])

  elif choices == 'Train Model':
    st.subheader("Train Machine Learning Models for Stock Prediction")
    st.markdown('')
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    stock_select = st.selectbox("", stocks, index=0)
    df1 = load_data(stock_select)
    df1 = df1.reset_index()
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    options = ['Select your option', 'Linear Regression','LSTM']
    st.markdown('')
    st.markdown('**_Select_ _Machine_ _Learning_ _Algorithms_ to Train**')
    models = st.selectbox("", options)
    submit = st.button('Train Model')

    if models == 'LSTM':
        st.markdown('')
        st.markdown('')
        st.markdown("**Select the _Number_ _of_ _epochs_ and _batch_ _size_ for _training_ form the following**")
        st.markdown('')
        epoch = st.slider("Epochs", 0, 300, step=1)
        b_s = st.slider("Batch Size", 32, 1024, step=1)
        if submit:
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df1[['Date','Close']])
            
            df1= df1['Close'].to_frame()
            df1['SMA3'] = df1['Close'].rolling(3).mean()
            st.write(df1)
            
            last_value = df1['SMA3'].iat[-1]
            st.subheader('For the next day ')
            st.subheader(NXT)
            st.subheader('Price of the stock would be')
            st.header(last_value)
     

            create_train_test_LSTM(df1, epoch, b_s, stock_select)


    if models == 'Linear Regression':
        if submit:
            st.write('**Your _final_ _dataframe_ _for_ Training**')
            st.write(df1[['Date','Close']])
            train_data, test_data = create_train_test_data(df1)
            pred_data = Linear_Regression_model(train_data, test_data)
            prediction_plot(pred_data, test_data, models, stock_select)
            df1= df1['Close'].to_frame()
            df1['SMA3'] = df1['Close'].rolling(3).mean()
            st.write(df1)
            
            last_value = df1['SMA3'].iat[-1]
            st.subheader('For the next day ')
            st.subheader(NXT)
            st.subheader('Price of the stock would be')
            st.header(last_value)




