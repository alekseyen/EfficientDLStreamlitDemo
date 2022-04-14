import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_feat_cols():
    feature = ['mean','std','min','max','qntl1','qntl3','median','rows','dow','hod','type']
    column = ['Converyer_Belt_Speed_m_min', 'Blower_Pressure_Bar','MatteTIn_Curent_Amp']

    feature_columns = []
    for i,ftr in enumerate(feature):
        feature_column = ''
        if i<7:
            for clmn in column:
                feature_column = f'{ftr}.{clmn}'
                feature_columns.append(feature_column)
        else :
            feature_column = f'{ftr}'
            feature_columns.append(feature_column)

    feature_columns = np.array(feature_columns)
    return feature_columns

def filter(x,y,xmin=60):
    include = True
    if x.shape[0]<xmin:
        include = False
    if y==0:
        include = False
    if (np.isnan(y)).any():
        include = False
    return include

def normalize(data_X,data_y,write,task,n):

    print("normalize")
    X_Scaler = MinMaxScaler()
    data_X = X_Scaler.fit_transform(data_X)
    Y_Scaler = MinMaxScaler()
    data_y = Y_Scaler.fit_transform(data_y)

    from pickle import dump
    if write:
        dump(X_Scaler, open(f'/var/www/app/X_scaler.pkl', 'wb'))
        dump(Y_Scaler, open(f'/var/www/app/y_scaler.pkl', 'wb'))

    return data_X,data_y,Y_Scaler,X_Scaler

def select_features(data_X,y,task,n):

    print("select_features")

    data_y = y.reshape(-1,1)

    data_X = np.asarray(data_X).astype('float32')
    data_y = np.asarray(data_y).astype('float32')

    cols = get_feat_cols()
    df_all = pd.DataFrame(data_X,columns=cols)

    fcolumn = np.load(f"/var/www/app/{task}/hours{n}/features.npy")[0]
    data_X = df_all[fcolumn]

    return data_X,data_y

def preprocess(x,y):

    dum = x
    x1 = np.mean(dum,axis=0)
    x2 = np.std(dum.astype('float'),axis=0)
    x3 = np.min(dum,axis=0)
    x4 = np.max(dum,axis=0)
    x5 = np.quantile(dum,q=0.25,axis=0)
    x6 = np.quantile(dum,q=0.75,axis=0)
    x7 = np.median(dum,axis=0)
    x8 = np.array([len(dum)])
    #x9 = np.array([y[0]])
    x10 = np.array([y[1]])
    x11 = np.array([y[2]])
    x12 = np.array([y[0]])
    x = np.concatenate((x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                        x8,
                        #x9,
                        x10,
                        x11,
                        x12),
                        axis=0)

    return np.array(x)

def transform(df_spc2,df_mc2,h,xmin,history,st_index):

    print("transform")
    X = []
    Y = []
    timestamp = []
    oneweek_onehour = timedelta(hours=h+2)
    for i in range (len(df_spc2)):

        date_index = datetime.strptime(st_index[i], "%Y-%m-%d %H:%M:%S")
        date_index_before = date_index - oneweek_onehour
        x = df_mc2[f'{date_index_before.strftime("%Y-%m-%d %H:%M:%S")}':f'{date_index.strftime("%Y-%m-%d %H:%M:%S")}']
        #x = x.iloc[:,1:]
        y = [df_spc2.iloc[i].MEANX,df_spc2.iloc[i]['type'],df_spc2.iloc[i].dayofweek,df_spc2.iloc[i].hour,df_spc2.index.values[i]]

        start_margin = timedelta(hours=h+history)
        start = date_index - start_margin
        end_margin = timedelta(hours=h)
        end = date_index - end_margin
        x_feature = x[f'{start.strftime("%Y-%m-%d %H:%M:%S")}':f'{end.strftime("%Y-%m-%d %H:%M:%S")}']
        x_feature.dropna(axis=0, how='any', inplace=True)

        include = filter(x_feature,y[:-1],xmin = xmin)
        if include:
            x_feature = preprocess(x_feature,y[1:])
            X.append(x_feature)
            Y.append(y[0])
            timestamp.append(y[-1])
    X = np.array(X)
    Y = np.array(Y)
    timestamp = np.array(timestamp)

    return X,Y

def create_dataset(df_spc2,df_mc2,n,write,xmin,history,st_index,task):
    X,y = transform(df_spc2=df_spc2,df_mc2=df_mc2,h=n,xmin=xmin,history=history,st_index=st_index)
    X,y = select_features(X,y,task,n)
    X,y,y_scaler,X_Scaler = normalize(X,y,write,task,n)

    return X,y,y_scaler,X_Scaler

def preprocess_dataset_st(df_mc,df_spc,hours,task):

    print("preprocess_dataset")

    df_spc = df_spc[df_spc['MACHINE']=='MECO004']

    df_spc = df_spc.sort_values(by='DATE_TIME',ascending=True)
    df_mc = df_mc.sort_values(by='date_time',ascending=True)

    df_spc.reset_index(inplace=True)
    df_spc.drop(['index'], axis = 1, inplace = True) 
    df_mc.reset_index(inplace=True)
    df_mc.drop(['index'], axis = 1, inplace = True) 

    df_spc2 = df_spc.copy()
    df_mc2 = df_mc.copy()

    #df_mc2.drop(['mc_id'], axis = 1, inplace = True) 

    if 'LSL' in df_spc2.columns:
        df_spc2['type'] = df_spc2['LSL']==200
        df_spc2 = df_spc2[['DATE_TIME','MEANX','type']]
    else:
        df_spc2 = df_spc2[['DATE_TIME','MEANX']]

    #df_spc2 = df_spc2[df_spc2['type']==False]

    df_spc2.reset_index(inplace=True)
    df_spc2.drop(['index'], axis = 1, inplace = True) 
    df_mc2.reset_index(inplace=True)
    df_mc2 = df_mc2[['date_time','Converyer_Belt_Speed_m_min','Blower_Pressure_Bar','MatteTIn_Curent_Amp']]

    df_spc3 = df_spc2.copy()
    df_spc3['DATE_TIME'] = pd.to_datetime(df_spc3['DATE_TIME'])
    df_spc3['dayofweek'] = df_spc3['DATE_TIME'].dt.dayofweek
    df_spc3['hour'] = df_spc3['DATE_TIME'].dt.hour
    df_spc2['dayofweek'] = df_spc3['dayofweek']
    df_spc2['hour'] = df_spc3['hour']

    df_mc2.set_index('date_time', inplace=True)
    df_spc2.set_index('DATE_TIME', inplace=True)
    st_index = df_spc2.index
    mc_index = df_mc2.index

    history = 1
    n = hours
    write = True
    xmin = 2
    X,y,y_scaler,X_Scaler = create_dataset(df_spc2,df_mc2,n,write,xmin,history,st_index,task)

    return X,y,y_scaler,X_Scaler
