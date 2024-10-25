# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:34:27 2024

@author: Dell
"""

import pickle
import streamlit as st

st.title('Bnakruptcy Prediction :bar_chart:')

load = open('C:/Users/Dell/Downloads/model (13).pkl','rb')
model = pickle.load(load)


def predict(ROA_C,Operating_Gross_Margin,Operating_Profit_Rate,Operating_Expense_Rate,RDER,Cash_flow_rate,OPPS,PSNPBT,RSGPGR,OPGR,Cash_Reinvestment,Current_Ratio,TD_TNW,Debt_ratio,Net_worth_Assets,Borrowing_dependency,OP_PC,NPBT_PC,NI_SE,Liability_to_Equity,DFL,ICR_IE_EBIT,Equity_to_Liability):
    prediction = model.predict([[ROA_C,Operating_Gross_Margin,Operating_Profit_Rate,Operating_Expense_Rate,RDER,Cash_flow_rate,OPPS,PSNPBT,RSGPGR,OPGR,Cash_Reinvestment,Current_Ratio,TD_TNW,Debt_ratio,Net_worth_Assets,Borrowing_dependency,OP_PC,NPBT_PC,NI_SE,Liability_to_Equity,DFL,ICR_IE_EBIT,Equity_to_Liability]])
    return prediction

def main():
    
    st.markdown('This is a prediction for bankruptcy :chart:')
    ROA_C = st.number_input('ROA_C',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Operating_Gross_Margin = st.number_input('Operating_Gross_Margin', min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Operating_Profit_Rate = st.number_input('Operating_Profit_Rate',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Operating_Expense_Rate = st.number_input('Operating_Expense_Rate',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    RDER = st.number_input('RDER',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Cash_flow_rate = st.number_input('Cash_flow_rate',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    OPPS = st.number_input('OPPS',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    PSNPBT = st.number_input('PSNPBT',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    RSGPGR = st.number_input('RSGPGR',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    OPGR = st.number_input('OPGR',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Cash_Reinvestment = st.number_input('Cash_Reinvestment',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Current_Ratio = st.number_input('Current_Ration',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    TD_TNW = st.number_input('TD_TNW',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Debt_ratio = st.number_input('Debt_ratio',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Net_worth_Assets = st.number_input('Net_worth_Assets',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Borrowing_dependency = st.number_input('Borrowing_dependency',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    OP_PC = st.number_input('OP_PC',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    NPBT_PC = st.number_input('NPBT_PC',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    NI_SE = st.number_input('NI_SE',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Liability_to_Equity = st.number_input('Liability_to_Equity',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    DFL = st.number_input('DFl',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    ICR_IE_EBIT = st.number_input('ICR_IE_EBIT',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    Equity_to_Liability = st.number_input('Equity_to_Liability',min_value=0.000000, value=0.000000, step=0.000001,format="%.6f")
    
    
    
    if st.button('Predict'):
        result = predict(ROA_C,Operating_Gross_Margin,Operating_Profit_Rate,Operating_Expense_Rate,RDER,Cash_flow_rate,OPPS,PSNPBT,RSGPGR,OPGR,Cash_Reinvestment,Current_Ratio,TD_TNW,Debt_ratio,Net_worth_Assets,Borrowing_dependency,OP_PC,NPBT_PC,NI_SE,Liability_to_Equity,DFL,ICR_IE_EBIT,Equity_to_Liability)
        if predict == 1:
            st.success("The company is likely to go bankrupt {}".format(result))
        else:
            st.success("The company is not likely to go bankrupt {}.".format(result))                      
                          
        
if __name__ == '__main__':
    main()
