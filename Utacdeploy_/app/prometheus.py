from prometheus_client import Gauge, Counter, Enum

class PROM():

    def __init__(self):

        #Prediction Info
        self.REQUEST_PREDICTION_CT3hours = Gauge(
            'request_prediction_ct3hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_CT24hours = Gauge(
            'request_prediction_ct24hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_CT168hours = Gauge(
            'request_prediction_ct168hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )

        self.REQUEST_PREDICTION_ST3hours = Gauge(
            'request_prediction_st3hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_ST24hours = Gauge(
            'request_prediction_st24hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )
        self.REQUEST_PREDICTION_ST168hours = Gauge(
            'request_prediction_st168hours', 'Prediction Result',
            ['app_name', 'method', 'endpoint']
        )

        #Model Version Info
        self.MODEL_VERSION_CT3hours = Gauge(
            'model_version_ct3hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_CT24hours = Gauge(
            'model_version_ct24hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_CT168hours = Gauge(
            'model_version_ct168hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )

        self.MODEL_VERSION_ST3hours = Gauge(
            'model_version_st3hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_ST24hours = Gauge(
            'model_version_st24hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_ST168hours = Gauge(
            'model_version_st168hours', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )

        #Model Retrain Total Info
        self.MODEL_RETRAIN_CT3hours = Counter(
            'model_retrain_ct3hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_CT24hours = Counter(
            'model_retrain_ct24hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_CT168hours = Counter(
            'model_retrain_ct168hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )

        self.MODEL_RETRAIN_ST3hours = Counter(
            'model_retrain_st3hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_ST24hours = Counter(
            'model_retrain_st24hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_ST168hours = Counter(
            'model_retrain_st168hours', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )

        #Model Retraining Status
        self.MODEL_STATUS_CT3hours = Enum(
            'model_status_ct3hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_CT24hours = Enum(
            'model_status_ct24hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_CT168hours = Enum(
            'model_status_ct168hours', 'Model Status',
            states=['idle', 'retraining']
        )

        self.MODEL_STATUS_ST3hours = Enum(
            'model_status_st3hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_ST24hours = Enum(
            'model_status_st24hours', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_ST168hours = Enum(
            'model_status_st168hours', 'Model Status',
            states=['idle', 'retraining']
        )


        ### Alarm & Defect
        
        #Prediction Info
        self.REQUEST_PREDICTION_NO_ALARM = Gauge(
            'request_no_alarm_prediction', 'No Alarm Prediction'
        )
        self.REQUEST_PREDICTION_INLOAD_PROB = Gauge(
            'request_inload_prob_prediction', 'INLOAD PROB*** Prediction'
        )
        self.REQUEST_PREDICTION_OTHERS = Gauge(
            'request_others_prediction', 'Others Prediction'
        )
        self.REQUEST_PREDICTION_OTHERS_MACHINE_PROBLEM = Gauge(
            'request_others_machine_problem_prediction', 'Others Machine Problem*** Prediction'
        )
        self.REQUEST_PREDICTION_RECTIFIERS_PROBLEM = Gauge(
            'request_rectifiers_problem_prediction', 'Rectifiers problem*** Prediction'
        )
        self.REQUEST_PREDICTION_UNLOAD_PROB = Gauge(
            'request_unload_prob_prediction', 'UNLOAD PROB*** Prediction'
        )
        
        self.REQUEST_PREDICTION_NO_DEFECT = Gauge(
            'request_no_defect_prediction', 'No Defect Prediction'
        )
        self.REQUEST_PREDICTION_COPPER_RESIDUE_GU = Gauge(
            'request_copper_residue_gu_prediction', 'COPPER RESIDUE_GU Prediction'
        )
        self.REQUEST_PREDICTION_EXPOSED_FOREIGN_MATERIAL_DOT_GT = Gauge(
            'request_exposed_foreign_material_dot_gt_prediction', 'EXPOSED FOREIGN MATERIAL (DOT)_GT Prediction'
        )
        self.REQUEST_PREDICTION_FLASH_RESIN_BLEED_SM = Gauge(
            'request_flash_resin_bleed_sm_prediction', 'FLASH / RESIN BLEED_SM Prediction'
        )
        self.REQUEST_PREDICTION_OTHERS_DEFECT = Gauge(
            'request_others_defect_prediction', 'Others Prediction'
        )
        
        #Model Version Info
        self.MODEL_VERSION_AlarmBinary = Gauge(
            'model_version_alarm_binary', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_AlarmMulticlass = Gauge(
            'model_version_alarm_multiclass', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )

        self.MODEL_VERSION_DefectBinary = Gauge(
            'model_version_defect_binary', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_VERSION_DefectMulticlass = Gauge(
            'model_version_defect_multiclass', 'Model Version',
            ['app_name', 'method', 'endpoint']
        )

        #Model Retrain Total Info
        self.MODEL_RETRAIN_AlarmBinary = Counter(
            'model_retrain_alarm_binary', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_AlarmMulticlass = Counter(
            'model_retrain_alarm_multiclass', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )

        self.MODEL_RETRAIN_DefectBinary = Counter(
            'model_retrain_defect_binary', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        self.MODEL_RETRAIN_DefectMulticlass = Counter(
            'model_retrain_defect_multiclass', 'Model Retrain',
            ['app_name', 'method', 'endpoint']
        )
        
        #Model Retraining Status
        self.MODEL_STATUS_AlarmBinary = Enum(
            'model_status_alarm_binary', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_AlarmMulticlass = Enum(
            'model_status_alarm_multiclass', 'Model Status',
            states=['idle', 'retraining']
        )

        self.MODEL_STATUS_DefectBinary = Enum(
            'model_status_defect_binary', 'Model Status',
            states=['idle', 'retraining']
        )
        self.MODEL_STATUS_DefectMulticlass = Enum(
            'model_status_defect_multiclass', 'Model Status',
            states=['idle', 'retraining']
        )