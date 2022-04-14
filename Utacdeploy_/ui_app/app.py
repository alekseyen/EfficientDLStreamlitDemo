import streamlit as st
import pandas as pd
import requests
import os

ip = os.environ['RETRAIN_IP']

st.set_page_config(layout="wide")
st.write("""
# Model Retraining Interface
""")

st.sidebar.header('User Input')
retraining_mode = st.sidebar.selectbox('Retraining Type',("all model","specific model"))

# Collects user input features into dataframe
if retraining_mode=='all model':
    mc_data = st.sidebar.file_uploader("Upload Machine Data", type=["csv"])
    ct_data = st.sidebar.file_uploader("Upload Chemical Tin Data", type=["csv"])
    st_data = st.sidebar.file_uploader("Upload Solder Thickness Data", type=["csv"])
    alarm_data = st.sidebar.file_uploader("Upload Alarm Data", type=["csv"])
    defect_data = st.sidebar.file_uploader("Upload Defect Data", type=["csv"])
    epoch = st.sidebar.number_input("epochs",step=1)

    if mc_data is not None:
        mc_data = pd.read_csv(mc_data)
        mc_data.to_csv("mc.csv",index=False)
        st.text('Machine Data Summary')
        st.dataframe(mc_data.describe())
        #minioClient.fput_object("dataset","mc.csv","mc.csv")
    if ct_data is not None:
        ct_data = pd.read_csv(ct_data)
        ct_data.to_csv("ct.csv",index=False)
        st.text('Chemical Tin Data Summary')
        st.dataframe(ct_data.describe())
        #minioClient.fput_object("dataset","ct.csv","ct.csv")
    if st_data is not None:
        st_data = pd.read_csv(st_data)
        st_data.to_csv("st.csv",index=False)
        st.text('Solder Thickness Data Summary')
        st.dataframe(st_data.describe())
        #minioClient.fput_object("dataset","st.csv","st.csv")
    if alarm_data is not None:
        alarm_data = pd.read_csv(alarm_data)
        alarm_data.to_csv("alarm.csv",index=False)
        st.text(f'Alarm Data Summary')
        st.dataframe(alarm_data.describe())
    if defect_data is not None:
        defect_data = pd.read_csv(defect_data)
        defect_data.to_csv("defect.csv",index=False)
        st.text(f'Defect Data Summary')
        st.dataframe(defect_data.describe())

    if st.sidebar.button("Start Retrain"):
        url = 'http://nginx:81/retrain_model_all_streamlit'
        headers = {'Content-Type': 'application/json'}

        obj = {"epoch":int(epoch),
                "mc_path":"mc.csv",
                "ct_path":"ct.csv",
                "st_path":"st.csv",
                "alarm_path": "alarm.csv",
                "defect_path": "defect.csv"
                }
                
        x = requests.post(url, json = obj, headers=headers)
        st.info(f'Model Retraining Started, See the progress [here](http://{ip}:3000)')

elif retraining_mode=='specific model':
    model_name = st.sidebar.selectbox('Model Name',('chemical_tin','solder_thickness','alarm','defect'))
    
    if model_name == 'chemical_tin' or model_name == 'solder_thickness':
        model_type = st.sidebar.selectbox('Model Type',(3,24,168))
    else:
        model_type = st.sidebar.selectbox('Model Type',('binary', 'multiclass'))

    mc_data = st.sidebar.file_uploader("Upload Machine Data", type=["csv"])
    
    if model_name == 'chemical_tin' or model_name == 'solder_thickness':
        spc_data = st.sidebar.file_uploader("Upload Chemical Tin Data", type=["csv"]) if model_name=='chemical_tin' else st.sidebar.file_uploader("Upload Solder Thickness Data", type=["csv"])
        epoch = st.sidebar.number_input("epochs",step=1)
    elif model_name == 'alarm':
        alarm_data = st.sidebar.file_uploader("Upload Alarm Data", type=["csv"])
    elif model_name == 'defect':
        defect_data = st.sidebar.file_uploader("Upload Defect Data", type=["csv"])
        

    if mc_data is not None:
        mc_data = pd.read_csv(mc_data)
        mc_data.to_csv("mc.csv",index=False)
        st.text('Machine Data Summary')
        st.dataframe(mc_data.describe())
        #minioClient.fput_object("dataset","mc.csv","mc.csv")
    if model_name == 'chemical_tin' or model_name == 'solder_thickness':
        if spc_data is not None:
            spc_data = pd.read_csv(spc_data)
            spc_data.to_csv("spc.csv",index=False)
            st.text(f'{model_name} Data Summary')
            st.dataframe(spc_data.describe())
            #minioClient.fput_object("dataset","spc.csv","spc.csv")
    elif model_name == 'alarm':
        if alarm_data is not None:
            alarm_data = pd.read_csv(alarm_data)
            alarm_data.to_csv("alarm.csv",index=False)
            st.text(f'Alarm Data Summary')
            st.dataframe(alarm_data.describe())
    elif model_name == 'defect':
        if defect_data is not None:
            defect_data = pd.read_csv(defect_data)
            defect_data.to_csv("defect.csv",index=False)
            st.text(f'Defect Data Summary')
            st.dataframe(defect_data.describe())

    if st.sidebar.button("Start Retrain"):
        if model_name == 'chemical_tin' or model_name == 'solder_thickness':
            url = 'http://nginx:81/retrain_model_streamlit'
            headers = {'Content-Type': 'application/json'}
            obj = {"model_name":model_name,
                    "hours":model_type,
                    "epoch":int(epoch),
                    "mc_path":"mc.csv",
                    "spc_path":"spc.csv"
                    }
        else:
            url = 'http://nginx:81/retrain_model_streamlit'
            headers = {'Content-Type': 'application/json'}
            obj = {"model_name":model_name+'_'+model_type,
                    "hours":1,
                    "mc_path":"mc.csv",
                    model_name+"_path": model_name+".csv"
                    }
        x = requests.post(url, json = obj, headers=headers)
        st.info(f'Model Retraining Started, See the progress [here](http://{ip}:3000)')