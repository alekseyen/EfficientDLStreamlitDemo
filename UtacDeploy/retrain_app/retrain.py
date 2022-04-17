from retrain_app.retrain_preprocess_st import preprocess_dataset_st
from retrain_app.retrain_preprocess_ct import preprocess_dataset_ct
from retrain_app.retrain_preprocess_alarm import binary_alarm_preprocess, multiclass_alarm_preprocess
from retrain_app.retrain_preprocess_defect import binary_defect_preprocess, multiclass_defect_preprocess
from retrain_app.retrain_model import model_retrain, change_production_model_to, classification_model_retrain
from retrain_app.retrain_parameters import PARAM
from retrain_app.minio_client import MINIO
import prometheus_client
from flask import Flask, request, jsonify, Response
import requests
import pandas as pd

app = Flask(__name__)

print("creating minio client")
minio = MINIO()
minioClient = minio.minioClient
found = minioClient.bucket_exists("mlflow")
if not found:
    minioClient.make_bucket("mlflow")
print("minio client created")

print("setting up prometheus endpoint")
@app.route('/metrics')
def metrics():
    CONTENT_TYPE_LATEST = str('text/plain; version=0.0.4; charset=utf-8')
    return Response(prometheus_client.generate_latest(), mimetype=CONTENT_TYPE_LATEST)
print("prometheus endpoint created")

#parameters for retraining
param = PARAM()

@app.route('/retrain_model', methods=['POST'])
def retrain_model():    
    context = request.get_json()
    param.task = context['model_name']
    
    if param.task == 'chemical_tin' or param.task == 'solder_thickness':
        param.hours = context['hours']
        param.epoch = context['epoch']
        param.mc_path = context['mc_path']
        param.spc_path = context['spc_path']
    
    elif param.task == 'alarm_binary' or param.task == 'alarm_multiclass':
        param.mc_path = context['mc_path']
        param.alarm_path = context['alarm_path']
        param.hours = context['hours']
    
    elif param.task == 'defect_binary' or param.task == 'defect_multiclass':
        param.mc_path = context['mc_path']
        param.defect_path = context['defect_path']
        param.hours = context['hours']
    
    response = Response(f'retraining {param.task}_{param.hours} model started')
    
    @response.call_on_close
    def on_close():
        
        if param.task == 'chemical_tin' or param.task == 'solder_thickness':
            minioClient.fget_object('dataset',
                                    param.mc_path,
                                    "df_mc.csv"
                                    )
            minioClient.fget_object('dataset',
                                    param.spc_path,
                                    "df_spc.csv"
                                    )

            df_mc = pd.read_csv("df_mc.csv")
            df_spc = pd.read_csv("df_spc.csv")

            url = 'http://nginx:80/update_training_status'
            headers = {'Content-Type': 'application/json'}

            obj = {"task":param.task,
                "hours":param.hours,
                "status":"retraining"
                }
            x = requests.post(url, json = obj, headers=headers)

            X,y,y_scaler,X_Scaler = preprocess_dataset_st(df_mc,df_spc,param.hours,param.task) if param.task=="solder_thickness" else preprocess_dataset_ct(df_mc,df_spc,param.hours,param.task)
            success = model_retrain(X,y,y_scaler,X_Scaler,param.task,param.hours,param.epoch)
        
        elif param.task == 'alarm_binary' or param.task == 'alarm_multiclass':
            minioClient.fget_object('dataset',
                                    param.mc_path,
                                    "df_mc.csv"
                                    )
            minioClient.fget_object('dataset',
                                    param.alarm_path,
                                    "df_alarm.csv"
                                    )
            
            df_mc = pd.read_csv("df_mc.csv")
            df_alarm = pd.read_csv("df_alarm.csv")

            url = 'http://nginx:80/update_training_status'
            headers = {'Content-Type': 'application/json'}
            
            obj = {"task":param.task,
                "hours":param.hours,
                "status":"retraining"
                }
            x = requests.post(url, json = obj, headers=headers)

            if param.task == 'alarm_binary':
                X_train, y_train, X_test, y_test = binary_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)
            elif param.task == 'alarm_multiclass':
                X_train, y_train, X_test, y_test = multiclass_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)
        
        elif param.task == 'defect_binary' or param.task == 'defect_multiclass':
            minioClient.fget_object('dataset',
                                    param.mc_path,
                                    "df_mc.csv"
                                    )
            minioClient.fget_object('dataset',
                                    param.defect_path,
                                    "df_defect.csv"
                                    )
            
            df_mc = pd.read_csv("df_mc.csv")
            df_defect = pd.read_csv("df_defect.csv")

            url = 'http://nginx:80/update_training_status'
            headers = {'Content-Type': 'application/json'}
            
            obj = {"task":param.task,
                "hours":param.hours,
                "status":"retraining"
                }
            x = requests.post(url, json = obj, headers=headers)

            if param.task == 'defect_binary':
                X_train, y_train, X_test, y_test = binary_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)
            elif param.task == 'defect_multiclass':
                X_train, y_train, X_test, y_test = multiclass_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)

        obj["status"] = "idle"
        x = requests.post(url, json = obj, headers=headers)

        if success:   
            print("reinit resource")
            from app.resources import RSC
            global rsc
            rsc = RSC()
            print("new resource is initialized")

            url = 'http://nginx:80/update_model'
            headers = {'Content-Type': 'application/json'}
            obj = {"task":param.task,
                "hours":param.hours
                }
            x = requests.post(url, json = obj, headers=headers)

    try:
        return response
    except:
        return 'There is an issue'

@app.route('/retrain_model_all', methods=['POST'])
def retrain_model_all():
    context = request.get_json()
    param.epoch = context['epoch']
    param.mc_path = context['mc_path']
    param.ct_path = context['ct_path']
    param.st_path = context['st_path']
    
    param.alarm_path = context['alarm_path']
    param.defect_path = context['defect_path']
    
    response = Response('retraining all model started')
    
    @response.call_on_close
    def on_close():
        minioClient.fget_object('dataset',
                                param.mc_path,
                                "df_mc.csv"
                                )
        minioClient.fget_object('dataset',
                                param.ct_path,
                                "df_ct.csv"
                                )
        minioClient.fget_object('dataset',
                                param.st_path,
                                "df_st.csv"
                                )

        minioClient.fget_object('dataset',
                                param.alarm_path,
                                "df_alarm.csv"
                                )
        minioClient.fget_object('dataset',
                                param.defect_path,
                                "df_defect.csv"
                                )
        
        df_mc = pd.read_csv("df_mc.csv")
        df_ct = pd.read_csv("df_ct.csv")
        df_st = pd.read_csv("df_st.csv")
        
        df_alarm = pd.read_csv("df_alarm.csv")
        df_defect = pd.read_csv("df_defect.csv")

        tasks = ['chemical_tin','solder_thickness',
                'alarm_binary', 'alarm_multiclass', 
                'defect_binary', 'defect_multiclass']
        hours_list = [3,24,168]

        for task in tasks:
            if task=='chemical_tin' or task=='solder_thickness':
                for hours in hours_list:

                    url = 'http://nginx:80/update_training_status'
                    headers = {'Content-Type': 'application/json'}

                    obj = {"task":task,
                        "hours":hours,
                        "status":"retraining"
                        }
                    x = requests.post(url, json = obj, headers=headers)

                    X,y,y_scaler,X_Scaler = preprocess_dataset_st(df_mc,df_st,hours,task) if task=="solder_thickness" else preprocess_dataset_ct(df_mc,df_ct,hours,task)
                    success = model_retrain(X,y,y_scaler,X_Scaler,task,hours,param.epoch)

                    obj["status"] = "idle"
                    x = requests.post(url, json = obj, headers=headers)

                    if success:   
                        print("reinit resource")
                        from app.resources import RSC
                        global rsc
                        rsc = RSC()
                        print("new resource is initialized")

                        url = 'http://nginx:80/update_model'
                        headers = {'Content-Type': 'application/json'}
                        obj = {"task":task,
                            "hours":hours
                            }
                        x = requests.post(url, json = obj, headers=headers)
            
            elif task=='alarm_binary':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = binary_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)
            
            elif task=='alarm_multiclass':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = multiclass_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)

            elif task=='defect_binary':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = binary_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)
    
            elif task=='defect_multiclass':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = multiclass_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)

    try:
        return response
    except:
        return 'There is an issue'

@app.route('/change_production_model', methods=['POST'])
def change_production_model():
    context = request.get_json()
    task = context['model_name']
    hours = context['hours']
    version = context['version']

    success = change_production_model_to(task,hours,version)

    if success:   
        print("reinit resource")
        from app.resources import RSC
        global rsc
        rsc = RSC()
        print("new resource is initialized")

        url = 'http://nginx:80/update_model'
        headers = {'Content-Type': 'application/json'}
        obj = {"task":task,
                "hours":hours
                }
        x = requests.post(url, json = obj, headers=headers)

    try:
        return 'Success'
    except:
        return 'There is an issue'

@app.route('/retrain_model_streamlit', methods=['POST'])
def retrain_model_streamlit():
    context = request.get_json()
    param.task = context['model_name']
    
    if param.task == 'chemical_tin' or param.task == 'solder_thickness':
        param.hours = context['hours']
        param.epoch = context['epoch']
        param.mc_path = context['mc_path']
        param.spc_path = context['spc_path']
    
    elif param.task == 'alarm_binary' or param.task == 'alarm_multiclass':
        param.mc_path = context['mc_path']
        param.alarm_path = context['alarm_path']
        param.hours = context['hours']
    
    elif param.task == 'defect_binary' or param.task == 'defect_multiclass':
        param.mc_path = context['mc_path']
        param.defect_path = context['defect_path']
        param.hours = context['hours']
    
    response = Response(f'retraining {param.task}_{param.hours} model started')
    
    @response.call_on_close
    def on_close():

        if param.task == 'chemical_tin' or param.task == 'solder_thickness':
        
            df_mc = pd.read_csv("mc.csv")
            df_spc = pd.read_csv("spc.csv")

            url = 'http://nginx:80/update_training_status'
            headers = {'Content-Type': 'application/json'}

            obj = {"task":param.task,
                "hours":param.hours,
                "status":"retraining"
                }
            x = requests.post(url, json = obj, headers=headers)

            X,y,y_scaler,X_Scaler = preprocess_dataset_st(df_mc,df_spc,param.hours,param.task) if param.task=="solder_thickness" else preprocess_dataset_ct(df_mc,df_spc,param.hours,param.task)
            success = model_retrain(X,y,y_scaler,X_Scaler,param.task,param.hours,param.epoch)
        
        elif param.task == 'alarm_binary' or param.task == 'alarm_multiclass':
        
            df_mc = pd.read_csv("mc.csv")
            df_alarm = pd.read_csv("alarm.csv")

            url = 'http://nginx:80/update_training_status'
            headers = {'Content-Type': 'application/json'}

            obj = {"task":param.task,
                "hours":param.hours,
                "status":"retraining"
                }
            x = requests.post(url, json = obj, headers=headers)

            if param.task == 'alarm_binary':
                X_train, y_train, X_test, y_test = binary_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)
            elif param.task == 'alarm_multiclass':
                X_train, y_train, X_test, y_test = multiclass_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)

        elif param.task == 'defect_binary' or param.task == 'defect_multiclass':
        
            df_mc = pd.read_csv("mc.csv")
            df_defect = pd.read_csv("defect.csv")

            url = 'http://nginx:80/update_training_status'
            headers = {'Content-Type': 'application/json'}

            obj = {"task":param.task,
                "hours":param.hours,
                "status":"retraining"
                }
            x = requests.post(url, json = obj, headers=headers)

            if param.task == 'defect_binary':
                X_train, y_train, X_test, y_test = binary_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)
            elif param.task == 'defect_multiclass':
                X_train, y_train, X_test, y_test = multiclass_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test ,param.task, param.hours)
        
        obj["status"] = "idle"
        x = requests.post(url, json = obj, headers=headers)

        if success:   
            print("reinit resource")
            from app.resources import RSC
            global rsc
            rsc = RSC()
            print("new resource is initialized")

            url = 'http://nginx:80/update_model'
            headers = {'Content-Type': 'application/json'}
            obj = {"task":param.task,
                   "hours":param.hours
                   }
            x = requests.post(url, json = obj, headers=headers)

    try:
        return response
    except:
        return 'There is an issue'

@app.route('/retrain_model_all_streamlit', methods=['POST'])
def retrain_model_all_streamlit():
    context = request.get_json()
    param.epoch = context['epoch']
    param.mc_path = context['mc_path']
    param.ct_path = context['ct_path']
    param.st_path = context['st_path']
    
    param.alarm_path = context['alarm_path']
    param.defect_path = context['defect_path']
    
    response = Response('retraining all model started')
    
    @response.call_on_close
    def on_close():

        df_mc = pd.read_csv("mc.csv")
        df_ct = pd.read_csv("ct.csv")
        df_st = pd.read_csv("st.csv")
        
        df_alarm = pd.read_csv("alarm.csv")
        df_defect = pd.read_csv("defect.csv")

        tasks = ['chemical_tin','solder_thickness',
                'alarm_binary', 'alarm_multiclass', 
                'defect_binary', 'defect_multiclass']
        
        hours_list = [3,24,168]

        for task in tasks:
            
            if task=='chemical_tin' or task=='solder_thickness':
                for hours in hours_list:

                    url = 'http://nginx:80/update_training_status'
                    headers = {'Content-Type': 'application/json'}

                    obj = {"task":task,
                        "hours":hours,
                        "status":"retraining"
                        }
                    x = requests.post(url, json = obj, headers=headers)

                    X,y,y_scaler,X_Scaler = preprocess_dataset_st(df_mc,df_st,hours,task) if task=="solder_thickness" else preprocess_dataset_ct(df_mc,df_ct,hours,task)
                    success = model_retrain(X,y,y_scaler,X_Scaler,task,hours,param.epoch)

                    obj["status"] = "idle"
                    x = requests.post(url, json = obj, headers=headers)

                    if success:   
                        print("reinit resource")
                        from app.resources import RSC
                        global rsc
                        rsc = RSC()
                        print("new resource is initialized")

                        url = 'http://nginx:80/update_model'
                        headers = {'Content-Type': 'application/json'}
                        obj = {"task":task,
                            "hours":hours
                            }
                        x = requests.post(url, json = obj, headers=headers)
            
            elif task=='alarm_binary':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = binary_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)

            elif task=='alarm_multiclass':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = multiclass_alarm_preprocess(df_alarm, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)
            
            elif task=='defect_binary':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = binary_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)
            
            elif task=='defect_multiclass':
                hours = 1
                url = 'http://nginx:80/update_training_status'
                headers = {'Content-Type': 'application/json'}

                obj = {"task":task,
                    "hours":hours,
                    "status":"retraining"
                    }
                x = requests.post(url, json = obj, headers=headers)

                X_train, y_train, X_test, y_test = multiclass_defect_preprocess(df_defect, df_mc)
                success = classification_model_retrain(X_train, X_test, y_train, y_test, task, hours)

                obj["status"] = "idle"
                x = requests.post(url, json = obj, headers=headers)

                if success:   
                    print("reinit resource")
                    from app.resources import RSC
                    # global rsc
                    rsc = RSC()
                    print("new resource is initialized")

                    url = 'http://nginx:80/update_model'
                    headers = {'Content-Type': 'application/json'}
                    obj = {"task":task,
                        "hours":hours
                        }
                    x = requests.post(url, json = obj, headers=headers)          
                    
    try:
        return response
    except:
        return 'There is an issue'

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=8001,debug=False)