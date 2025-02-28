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
                        
    )
logo = Image.open(r'C:\Users\Dell\Desktop\stockpriceproject\Sourcecode\imh.png')

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