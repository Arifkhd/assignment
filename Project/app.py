# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:03:57 2024

@author: Dell
"""

import pickle
import streamlit as st


st.title('power generated prediction:bar_chart:')

load = open('C:/Users/Dell/Downloads/model (1).pkl','rb')
model = pickle.load(load)

def predict(distance_to_solar_noon,temperature,wind_direction,wind_speed,sky_cover,visibility,humidity,average_wind_speed_period,average_pressure_period):
    prediction = model.predict([[distance_to_solar_noon,temperature,wind_direction,wind_speed,sky_cover,visibility,humidity,average_wind_speed_period,average_pressure_period]])
    return prediction

def main():
    st.markdown('This is a very simple webapp for prediction of solar power in jules :chart:')
    distance_to_solar_noon = st.number_input('Distance_to_solar_noon',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    temperature	 = st.number_input('temperature', min_value= 0 , max_value=100)
    wind_direction = st.number_input('wind_direction', min_value= 0 , max_value=100)
    wind_speed = st.number_input('wind_speed', min_value=00.00, value=00.00,format="%.1f")
    sky_cover = st.number_input('sky_cover', min_value= 0 , max_value=100)
    visibility = st.number_input('visibility', min_value=00.00, value=00.00,format="%.1f")
    humidity = st.number_input('humidity', min_value= 0 , max_value=100)
    average_wind_speed_period = st.number_input('average_wind_speed_period', min_value=00.00, value=00.00,format="%.1f")
    average_pressure_period = st.number_input('average_pressure_period', min_value=00.00, value=00.00,format="%.2f")
    
    if st.button('Predict'):
        result = predict(distance_to_solar_noon,temperature,wind_direction,wind_speed,sky_cover,visibility,humidity,average_wind_speed_period,average_pressure_period)
        st.success('The power generated is : J{} '.format(result))                       
                          
        
if __name__ == '__main__':
    main()