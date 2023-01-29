import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.colors as colors
from sklearn.metrics import mean_absolute_error
import numpy as np

from PIL import Image
from io import BytesIO

import xgboost as xgb



pages_name = ['Presentation','Modelisation']
# Create a sidebar with a radio button to select the page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", pages_name)
#################

def fahrenheit_to_celsius(temp_fahrenheit):
    temp_celsius = (temp_fahrenheit - 32) * 5/9
    return temp_celsius
########################

df_lyon = pd.read_csv("dataset\Lyon.csv")
df_lyon['ville'] = 'Lyon'

df_paris = pd.read_csv("dataset\Paris.csv")
df_paris['ville'] = 'Paris'

df_madrid = pd.read_csv("dataset\Madrid.csv")
df_madrid['ville'] = 'Madrid'

df_marseille = pd.read_csv("dataset\Marseille.csv")
df_marseille['ville'] = 'Marseille'

df_milan = pd.read_csv("dataset\Milan.csv")
df_milan['ville'] = 'Milan'

df_new_york = pd.read_csv("dataset\\New_york.csv")
df_new_york['ville'] = 'New York'

df = pd.concat([df_lyon, df_paris, df_madrid, df_marseille, df_milan, df_new_york], axis=0)
df = df.drop(columns=['Unnamed: 0'])

selected_city = st.sidebar.selectbox("Sélectionnez une ville", df['ville'].unique())


########################
if page == 'Presentation':
    text = "Machine Learning Application for Weather Forecast"

    colored_text = ""
    colors = ["blue", "white", "red"]

    for i, letter in enumerate(text):
        color = colors[i % len(colors)]
        colored_text += f"<span style='color: {color}'>{letter}</span>"

    st.markdown(f"<h1 style='text-align: center; font-size:25px; border-radius:2%; background-color:black; padding: 10px; color:white;'>{colored_text}</h1>", unsafe_allow_html=True)


    st.markdown("<h1 style='text-align: center; color: black; font-size:25px; border-radius:2%; background-color:black; padding: 10px; color:white;'>Machine Learning Application for Weather Forecast</h1>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 2px solid #F2F2F2;'>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: white; font-size:20px;'>Team Members:</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: white; font-size:18px;'>ABABII Anisoara, LAAUMARI Oussama, ZAHI Camil </h3>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 2px solid #F2F2F2;'>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: white; font-size:20px;'>About the Application:</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: white; font-size:18px;'>We have developed a weather forecasting application that utilizes the XGBoost algorithm to predict the weather in certain cities that are dear to us. Our model may not be perfect, but it provides valuable insights and is constantly improving. We invite you to discover our application and see for yourself the potential it holds in terms of weather forecasting. Whether you live in one of the cities we cover or are planning a visit, our application will help you stay informed and prepared for the weather ahead..</h3>", unsafe_allow_html=True)

    #st.image('https://media.tenor.com/eZqKP1ThC2oAAAAC/todays-weather-weather.gif')
    
        
    if selected_city == "Paris":
        st.image("pictures\Paris.jpg")
    if selected_city == "Lyon":
            st.image("pictures\Lyon.jpg")
    elif selected_city == "Madrid":
        st.image("pictures\Madrid.jpg")
    elif selected_city == "Marseille":
        st.image("pictures\Marseille.jpg")
    elif selected_city == "Milan":
        st.image("pictures\Milan.jpg")
    elif selected_city == "New York":
        st.image("pictures\\NY.jpg")
    else:
        pass
    st.markdown("<h3 style='text-align: center; color: white; font-size:18px;'>"+ selected_city + "</h3>", unsafe_allow_html=True)

######################
    
# Charger votre dataframe de villes
#df = pd.read_csv("Lyon.csv")
#df['ville'] = 'Lyon'


# Créer une barre déroulante pour sélectionner une ville


    #image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg/417px-La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg"

   # response = requests.get(image_url)
   # img_paris = Image.open(BytesIO(response.content))
   # st.image(img_paris, width=400, caption="Paris", use_column_width=False)


elif page == 'Modelisation':

    filtered_df = df[df["ville"] == selected_city]
    st.markdown(f"# :blue[Météo] :white[Aujourd'hui à] :red[{selected_city}] :sunny: :umbrella: :snowflake:")

    df = df[df['ville']==selected_city].copy()
    df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
    #st.dataframe(filtered_df.drop(columns=['ville']).head())

            


    ###################### Modèle
    df['Temp_avg'] = df['Temp_avg'].astype(float)
    df['Temp_avg'] = df['Temp_avg'].apply(fahrenheit_to_celsius)
    train = df[:int(len(df)*0.8)]
    test = df[int(len(df)*0.8):]

    train.index = train['Date']
    test.index = test['Date']

    def create_features(df):
        """
        Create time series features based on time series index.
        """
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        return df

    train_t = create_features(train)
    test_t = create_features(test)

    FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year']
    TARGET = 'Temp_avg'

    X_train = train_t[FEATURES]
    y_train = train_t[TARGET]

    X_test = test_t[FEATURES]
    y_test = test_t[TARGET]


    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                        n_estimators=1000,
                        early_stopping_rounds=50,
                        objective='reg:linear',
                        max_depth=4,
                        learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)])



    test['prediction'] = reg.predict(X_test)
    df_total = train.merge(test[['prediction']], how='left', left_index=True, right_index=True)

    score = np.sqrt(mean_absolute_error(y_test, test['prediction']))
    st.write(f'En moyenne notre modèle fait une erreur de {score:0.2f} °C (MAE) en valeur absolue')

    def predict_with_date():
        today = datetime.now()
        date_array = [today.timetuple().tm_yday, today.weekday(), int((today.month -1) / 4) + 1, today.month, today.year]
        prediction = reg.predict(np.array(date_array).reshape(1,5))
        return prediction #fahrenheit_to_celsius(prediction)

    temp_pred = predict_with_date()

    fig1 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = temp_pred[0],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Temperature (°C) prédite selon notre modèle", 'font': {'size': 24}},
        gauge = {'axis': {'range': [-40, 40]},
                'bar': {'color': "red"},
                'steps' : [
                    {'range': [-40, -20], 'color': 'blue'},
                    {'range': [-20, 0], 'color': 'green'},
                    {'range': [0, 20], 'color': 'yellow'},
                    {'range': [20, 40], 'color': 'red'}
                ]}
    ))

    st.plotly_chart(fig1)
#############
    def predict_week(reg):
        today = datetime.now()
        predictions = []
        for i in range(7):
            date = today + timedelta(days=i)
            date_array = [date.timetuple().tm_yday, date.weekday(), int((date.month -1) / 4) + 1, date.month, date.year]
            prediction = reg.predict(np.array(date_array).reshape(1,5))
            predictions.append(prediction[0])
        df = pd.DataFrame({'Date': [today + timedelta(days=i) for i in range(7)],
                        'Prediction': predictions})
        df["Prediction"] = df["Prediction"].apply(lambda x: round(x,1))
        #df["text"] = df["Prediction"].apply(lambda x: "☀️" + str(x) if x > 25 else "☁️" + str(x) if x > 20 else "❄️" + str(x) if x < 10 else "🌤" + str(x))
        #df["Advice"] = df["Prediction"].apply(lambda x: float(x.split("°")[0]))
        df["Advice"] = df["Prediction"].apply(lambda x: "☃️ Il fait plus froid que dans un congélateur à légumes! Température : " + str(x) + "°C" if x < 0 
        else "❄️ N'oubliez pas de vous couvrir chaudement, il fait froid comme dans un frigo en panne! Température : " + str(x) + "°C" if x >= 0 and x < 5 
        else "🌬 Il fait frais, vous pouvez sortir sans risque de fondre comme une glace au soleil! Température : " + str(x) + "°C" if x >= 5 and x < 10 
        else "🌤 Il fait agréable, sortez profiter de cette belle journée! Température : " + str(x) + " °C" if x >= 10 and x < 20 
        else "🔥 Attention aux coups de chaud, n'oubliez pas de vous hydrater régulièrement! Température : " + str(x) + "°C" if x >= 20 and x < 30 
        else "☀️ Mieux vaut rester à l'ombre, c'est plus chaud que dans un sauna! Température : " + str(x) + "°C" if x >= 30 else "☀️ Il fait chaud, n'oubliez pas de vous hydrater et de vous protéger du soleil ! Température : " + str(x) + "°C")

        return df


    
    def plot_week_prediction(df):
        color_list = []
        for temp in df['Prediction']:
            if temp < 0:
                color_list.append("blue")
            elif temp < 5:
                color_list.append("lightblue")
            elif temp < 10:
                color_list.append("green")
            elif temp < 20:
                color_list.append("yellow")
            elif temp < 30:
                color_list.append("orange")
            else:
                color_list.append("red")

        fig = go.Figure(data=[go.Bar(
                x=df['Date'],
                y=df['Prediction'],
                #text=df["Advice"],
                hovertext=df["Advice"],
                marker_color=color_list
        )])
        fig.update_layout(title='Prévisions de température de la semaine selon notre modèle',
                        xaxis_title='Jour',
                        yaxis_title='Température',
                        )
        return fig
        
    
    week_pred = predict_week(reg)
    st.plotly_chart(plot_week_prediction(week_pred))
    st.write(week_pred["Advice"][0])
    #st.line_chart(week_pred, x='Date', y='Prediction')
   
    ##############################
    #st.title("Data Train/Test Split")
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True) 

    #fig, ax = plt.subplots(figsize=(15, 5))
    #train['Temp_avg'].plot(ax=ax, label='Training Set')
    #test['Temp_avg'].plot(ax=ax, label='Test Set')
    #ax.axvline(pd.to_datetime('2022-06-15'), color='black', ls='--')
    #ax.legend(['Training Set', 'Test Set'])
    #plt.xlim(left=pd.to_datetime('2023-01-27'))
    #st.pyplot(fig)
    ################
    api_key = 'f15d6887ef0449fa9f695040232701'

    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={selected_city}"
    response = requests.get(url)
    data = response.json()
    #st.write(data['current']['temp_c'])

    temp = data['current']['temp_c']

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = temp,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Temperature (°C) actuelle selon weatherapi", 'font': {'size': 24}},
        gauge = {'axis': {'range': [-40, 40]},
                'bar': {'color': "red"},
                'steps' : [
                    {'range': [-40, -20], 'color': 'blue'},
                    {'range': [-20, 0], 'color': 'green'},
                    {'range': [0, 20], 'color': 'yellow'},
                    {'range': [20, 40], 'color': 'red'}
                ]}
    ))

    st.plotly_chart(fig)

    def plot_week_prediction_api(df):
            color_list = []
            for temp in df['Temperature']:
                if temp < 0:
                    color_list.append("blue")
                elif temp < 5:
                    color_list.append("lightblue")
                elif temp < 10:
                    color_list.append("green")
                elif temp < 20:
                    color_list.append("yellow")
                elif temp < 30:
                    color_list.append("orange")
                else:
                    color_list.append("red")

            fig = go.Figure(data=[go.Bar(
                    x=df['Date'],
                    y=df['Temperature'],
                    #text=df["Advice"],
                    hovertext=df["Advice"],
                    marker_color=color_list
            )])
            fig.update_layout(title='Prévisions de température de la semaine selon weatherapi',
                            xaxis_title='Jour',
                            yaxis_title='Température',
                            )
            return fig


    

    def weather_week(selected_city):
        city = selected_city
        api_key = 'f15d6887ef0449fa9f695040232701'
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=7"
        response = requests.get(url)
        data = response.json()
        today = datetime.now()
        predictions = []
        for i in range(7):
            date = today + timedelta(days=i)
            temperature = data['forecast']['forecastday'][i]['day']['avgtemp_c']
            predictions.append(temperature)
        df = pd.DataFrame({'Date': [today + timedelta(days=i) for i in range(7)],
                            'Temperature': predictions})
        df["Advice"] = df["Temperature"].apply(lambda x: "☃️ Il fait plus froid que dans un congélateur à légumes! Température : " + str(x) + "°C" if x < 0 
            else "❄️ N'oubliez pas de vous couvrir chaudement, il fait froid comme dans un frigo en panne! Température : " + str(x) + "°C" if x >= 0 and x < 5 
            else "🌬 Il fait frais, vous pouvez sortir sans risque de fondre comme une glace au soleil! Température : " + str(x) + "°C" if x >= 5 and x < 10 
            else "🌤 Il fait agréable, sortez profiter de cette belle journée! Température : " + str(x) + " °C" if x >= 10 and x < 20 
            else "🔥 Attention aux coups de chaud, n'oubliez pas de vous hydrater régulièrement! Température : " + str(x) + "°C" if x >= 20 and x < 30 
            else "☀️ Mieux vaut rester à l'ombre, c'est plus chaud que dans un sauna! Température : " + str(x) + "°C" if x >= 30 else "☀️ Il fait chaud, n'oubliez pas de vous hydrater et de vous protéger du soleil ! Température : " + str(x) + "°C")
        return df


    fig4 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = weather_week(selected_city)['Temperature'][0],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Temperature (°C) prédite selon le modèle de weatherapi", 'font': {'size': 24}},
        gauge = {'axis': {'range': [-40, 40]},
                'bar': {'color': "red"},
                'steps' : [
                    {'range': [-40, -20], 'color': 'blue'},
                    {'range': [-20, 0], 'color': 'green'},
                    {'range': [0, 20], 'color': 'yellow'},
                    {'range': [20, 40], 'color': 'red'}
                ]}
    ))
    st.plotly_chart(fig4)


    week_pred_api = weather_week(selected_city)
    st.plotly_chart(plot_week_prediction_api(week_pred_api))
    st.write(week_pred_api["Advice"][0])    
    ####

    #icon_url = "http:" + data['current']['condition']['icon']
   # response = requests.get(icon_url)
    #img1 = Image.open(BytesIO(response.content))
    #st.image(img1)


#####


