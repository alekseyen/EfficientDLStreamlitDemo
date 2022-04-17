import onnxruntime as rt
import numpy as np
from pickle import load
import os
import mlflow
import onnx
import joblib
from retrain_app.retrain_model import use_retrained_model
from retrain_app.minio_client import MINIO
class RSC:

    def __init__(self,is_first=False):

        if is_first:

            minio = MINIO()
            minioClient = minio.minioClient
            found = minioClient.bucket_exists("mlflow")
            if not found:
                minioClient.make_bucket("mlflow")

            ip = os.environ['RETRAIN_IP']
            os.environ["AWS_ACCESS_KEY_ID"] = "minio"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{ip}:9000"
            mlflow.set_tracking_uri(f'http://{ip}:5000')
            
            tasks = ['chemical_tin','solder_thickness',
                    'alarm_binary', 'alarm_multiclass', 
                    'defect_binary', 'defect_multiclass']
            hours = [3,24,168]

            for task in tasks:
                
                if task=='chemical_tin' or task=='solder_thickness':
                    for hour in hours:
                        mlflow.set_experiment(f'{task}_{hour}')
                        with mlflow.start_run():
                            onnx_model = onnx.load(f"/var/www/app/{task}/hours{hour}/model.onnx")
                            mlflow.onnx.log_model(onnx_model, "onnx_model")
                            mlflow.log_artifact(f'/var/www/app/{task}/hours{hour}/X_scaler.pkl')
                            mlflow.log_artifact(f'/var/www/app/{task}/hours{hour}/y_scaler.pkl')
                            mlflow.log_artifact(f'/var/www/app/{task}/hours{hour}/features.npy')
                            mlflow.log_artifact(f'/var/www/app/{task}/hours{hour}/top5.npy')
                            run_id = mlflow.active_run().info.run_id
                        use_retrained_model('Production',f'{task}',hour,run_id)
                
                else:
                    hour = 1
                    mlflow.set_experiment(f'{task}_{hour}')
                    with mlflow.start_run():
                        model = joblib.load('/var/www/app/{}/{}/model.sav'.format(task.split('_', 1)[0], task.split('_', 1)[1]))
                        mlflow.sklearn.log_model(model, "model")
                        mlflow.log_artifact('/var/www/app/{}/{}/features.save'.format(task.split('_', 1)[0], task.split('_', 1)[1]))
                        mlflow.log_artifact('/var/www/app/{}/{}/top5.npy'.format(task.split('_', 1)[0], task.split('_', 1)[1]))
                        run_id = mlflow.active_run().info.run_id
                    use_retrained_model('Production',f'{task}',hour,run_id)

        self.model_ct3hours = rt.InferenceSession("/var/www/app/chemical_tin/hours3/model.onnx")
        self.X_scaler_ct3hours = load(open("/var/www/app/chemical_tin/hours3/X_scaler.pkl",'rb'))
        self.Y_scaler_ct3hours = load(open("/var/www/app/chemical_tin/hours3/y_scaler.pkl",'rb'))
        self.features_ct3hours = np.load("/var/www/app/chemical_tin/hours3/features.npy")[0]
        self.top5_ct3hours = np.load("/var/www/app/chemical_tin/hours3/top5.npy",allow_pickle=True)

        self.model_ct24hours = rt.InferenceSession("/var/www/app/chemical_tin/hours24/model.onnx")
        self.X_scaler_ct24hours = load(open("/var/www/app/chemical_tin/hours24/X_scaler.pkl",'rb'))
        self.Y_scaler_ct24hours = load(open("/var/www/app/chemical_tin/hours24/y_scaler.pkl",'rb'))
        self.features_ct24hours = np.load("/var/www/app/chemical_tin/hours24/features.npy")[0]
        self.top5_ct24hours = np.load("/var/www/app/chemical_tin/hours24/top5.npy",allow_pickle=True)

        self.model_ct168hours = rt.InferenceSession("/var/www/app/chemical_tin/hours168/model.onnx")
        self.X_scaler_ct168hours = load(open("/var/www/app/chemical_tin/hours168/X_scaler.pkl",'rb'))
        self.Y_scaler_ct168hours = load(open("/var/www/app/chemical_tin/hours168/y_scaler.pkl",'rb'))
        self.features_ct168hours = np.load("/var/www/app/chemical_tin/hours168/features.npy")[0]
        self.top5_ct168hours = np.load("/var/www/app/chemical_tin/hours168/top5.npy",allow_pickle=True)

        self.model_st3hours = rt.InferenceSession("/var/www/app/solder_thickness/hours3/model.onnx")
        self.X_scaler_st3hours = load(open("/var/www/app/solder_thickness/hours3/X_scaler.pkl",'rb'))
        self.Y_scaler_st3hours = load(open("/var/www/app/solder_thickness/hours3/y_scaler.pkl",'rb'))
        self.features_st3hours = np.load("/var/www/app/solder_thickness/hours3/features.npy")[0]
        self.top5_st3hours = np.load("/var/www/app/solder_thickness/hours3/top5.npy",allow_pickle=True)

        self.model_st24hours = rt.InferenceSession("/var/www/app/solder_thickness/hours24/model.onnx")
        self.X_scaler_st24hours = load(open("/var/www/app/solder_thickness/hours24/X_scaler.pkl",'rb'))
        self.Y_scaler_st24hours = load(open("/var/www/app/solder_thickness/hours24/y_scaler.pkl",'rb'))
        self.features_st24hours = np.load("/var/www/app/solder_thickness/hours24/features.npy")[0]
        self.top5_st24hours = np.load("/var/www/app/solder_thickness/hours24/top5.npy",allow_pickle=True)

        self.model_st168hours = rt.InferenceSession("/var/www/app/solder_thickness/hours168/model.onnx")
        self.X_scaler_st168hours = load(open("/var/www/app/solder_thickness/hours168/X_scaler.pkl",'rb'))
        self.Y_scaler_st168hours = load(open("/var/www/app/solder_thickness/hours168/y_scaler.pkl",'rb'))
        self.features_st168hours = np.load("/var/www/app/solder_thickness/hours168/features.npy")[0]
        self.top5_st168hours = np.load("/var/www/app/solder_thickness/hours168/top5.npy",allow_pickle=True)
        
        ### Alarm & Defect
    
        # alarm
        self.model_alarmbinary1hour = joblib.load("/var/www/app/alarm/binary/model.sav")
        self.features_alarmbinary1hour = joblib.load("/var/www/app/alarm/binary/features.save") 
        self.top5_alarmbinary1hour = np.load("/var/www/app/alarm/binary/top5.npy")
        
        self.model_alarmmulti1hour = joblib.load("/var/www/app/alarm/multiclass/model.sav")
        self.features_alarmulti1hour = joblib.load("/var/www/app/alarm/multiclass/features.save") 
        self.top5_alarmulti1hour = np.load("/var/www/app/alarm/multiclass/top5.npy")
        
        # defect
        self.model_defectbinary1hour = joblib.load("/var/www/app/defect/binary/model.sav")
        self.features_defectbinary1hour = joblib.load("/var/www/app/defect/binary/features.save") 
        self.top5_defectbinary1hour = np.load("/var/www/app/defect/binary/top5.npy")
        
        self.model_defectmulti1hour = joblib.load("/var/www/app/defect/multiclass/model.sav")
        self.features_defectmulti1hour = joblib.load("/var/www/app/defect/multiclass/features.save")
        self.top5_defectmulti1hour = np.load("/var/www/app/defect/multiclass/top5.npy")