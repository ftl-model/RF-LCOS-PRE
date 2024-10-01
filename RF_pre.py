###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('RF_pre.pkl')
scaler = joblib.load('scaler.pkl') 

# Define feature names
feature_names = [ "age", " preop_ucg_EF ", "BNP ", " DBIL ", "T3", "UREA"]

## Streamlit user interface
st.title("LCOS preoperative model")

# age: numerical input
age = st.number_input("Age:", min_value=18, max_value=100, value=60)
# preop_ucg_EF: numerical input
preop_ucg_EF = st.number_input("preop_ucg_EF:", min_value=10, max_value=100, value=60)
# BNP: numerical input
BNP = st.number_input("NT-proBNP（ng/L）:", min_value=100.00, max_value=50000.00, value=125.00)
# DBIL：numerical input
DBIL = st.number_input("DBIL（μmol/L) :", min_value=0.00, max_value=100.00, value=5.00)
# T3：numerical input
T3 =  st.number_input("T3（nmol/L ):", min_value=0.00, max_value=10.00, value=3.00)
# UREA: numerical input
UREA = st.number_input("UREA（mmol/L）:", min_value=0.00, max_value=50.00, value=5.50)


# Process inputs and make predictions
feature_values = [ age, preop_ucg_EF, BNP, DBIL ,T3, UREA]
features = np.array([feature_values])

if st.button("Predict"):    
    # 标准化特征
    standardized_features = scaler.transform(features)

    # Predict class and probabilities    
    predicted_class = model.predict(standardized_features)[0]   
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(1: Disease, 0: No Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
         advice = (            
                f"According to our model, you have a high risk of heart disease. "            
                f"The model predicts that your probability of having heart disease is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model, you have a low risk of heart disease. "            
                f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

# SHAP Explanation    
    st.subheader("SHAP Force Plot Explanation")    
    explainer_shap = shap.TreeExplainer(model)    
    shap_values = explainer_shap.shap_values(pd.DataFrame(standardized_features, columns=feature_names))    
# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame(standardized_features, columns=feature_names), matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame(standardized_features, columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')