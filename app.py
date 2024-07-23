import joblib
import streamlit as st
import numpy as np
import sklearn
import pandas as pd 

feature_scaler = joblib.load(r'feature_scaler.joblib')
label_scaler = joblib.load(r'label_scaler.joblib')

rf_model = joblib.load(r'rf_model.joblib')
elastic_model = joblib.load(r'elastic_model.joblib')
lasso_model = joblib.load(r'lasso_model.joblib')

importance_df = pd.read_csv(r'model_importances.csv')

st.title('Regression Models for PME/PMH')

housing_permits = st.text_input('Housing Permits Issued Per Year')
plug_in_hybrid = st.text_input('Plug-in EV Units Sold Per Year')
electric = st.text_input('EV Units Sold Per Year')
comm_construction = st.text_input('Commercial Construction Spending Per Year(millions)')
capex_iou = st.text_input('Underground CAPEX IOU')
model_type = st.selectbox('Choose Model Type', ['Random Forest', 'Elastic Net', 'Lasso'])

if model_type == 'Random Forest':
    model = rf_model
elif model_type == 'Elastic Net':
    model = elastic_model
elif model_type == 'Lasso':
    model = lasso_model

def infer(model, housing_permits, plug_in_hybrid, electric, comm_construction, capex_iou):
 
    sample = np.array([[housing_permits, plug_in_hybrid, electric, comm_construction, capex_iou]], dtype=float).reshape(1, -1)
    scaled_sample = feature_scaler.transform(sample)
    prediction = model.predict(scaled_sample)
    prediction = label_scaler.inverse_transform(prediction.reshape(-1, 1))
    st.success(f'PMG/PMH Yearly Income Prediction: ${format(round(prediction[0][0]), ",")}')

if st.button('Model Prediction'):
    infer(model, housing_permits, plug_in_hybrid, electric, comm_construction, capex_iou)

st.dataframe(importance_df)

