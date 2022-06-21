# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:10:15 2022

@author: AMD
"""


import pickle
import os
import numpy as np
import streamlit as st

MODEl_PATH = os.path.join(os.getcwd(),'model','best_pipeline.pkl')
with open(MODEl_PATH,'rb') as file:
    model = pickle.load(file)


#%% to test

with st.form("my_form"):
    age = st.number_input('Age')
#    sex = st.number_input('sex')
    cp = st.number_input('chest pain type')
    trtbps = st.number_input('trtbps')
#    chol = st.number_input('resting electrocardiographic results')
#    fbs = st.number_input('(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)')
#    restecg = st.number_input('restecg')
    thalachh = st.number_input('Maximum heart rate achieved')
    exng = st.number_input('exng')
    oldpeak = st.number_input('oldpeak')
#    slp = st.number_input('slp')
    caa = st.number_input('caa')
    thall = st.number_input('thalassemia')


    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [age,cp,trtbps,thalachh,exng,oldpeak,caa,thall]
        outcome = model.predict(np.expand_dims(np.array(X_new),axis = 0))
        st.write('The output is: ',(outcome)[0])