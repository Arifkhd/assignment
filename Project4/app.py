# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:32:02 2024

@author: Dell
"""

import pickle
import streamlit as st


st.title('power generated prediction:bar_chart:')

load = open('C:/Users/Dell/Downloads/mo.pkl','rb')
model = pickle.load(load)

def predict(Spend,Administration,Marketing_Spend,State):
    prediction = model.predict([[Spend,Administration,Marketing_Spend,State]])
    return prediction

def main():
    st.markdown('This is a very simple webapp for prediction of solar power in jules :chart:')
    Spend = st.number_input('Spend' ,min_value=0, max_value=1000000)
    Administration	 = st.number_input('Administration', min_value= 0 , max_value=1000000)
    Marketing_Spend = st.number_input('Marketing_spend', min_value= 0 , max_value=1000000)
    State = st.selectbox('State',('New York','California','Florida'))

    
    if st.button('Predict'):
        result = predict(Spend,Administration,Marketing_Spend,State)
        st.success('The profit of the company : ${} '.format(result))                       
                          
        
if __name__ == '__main__':
    main()