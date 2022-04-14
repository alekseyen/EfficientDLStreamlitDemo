import pandas as pd
import sklearn

def select_type_mc(df_alarm, df_mc, mc_type='MECO-04'):
    df_alarm_mc_type = df_alarm.loc[df_alarm.site == mc_type]
    df_alarm_mc_type['dt_start_down'] =  pd.to_datetime(df_alarm_mc_type['dt_start_down'])
    df_alarm_mc_type.sort_values('dt_start_down', inplace=True)
    df_alarm_mc_type.reset_index(drop=True, inplace=True)
    df_mc_mc_type = df_mc.copy()
    df_mc_mc_type['date_time'] =  pd.to_datetime(df_mc_mc_type['date_time'])
    df_mc_mc_type.sort_values('date_time', inplace=True)
    df_mc_mc_type.reset_index(drop=True, inplace=True)
    return df_alarm_mc_type, df_mc_mc_type

def select_data_with_alarm(df_alarm, alarms_list, hist_data_length=3, predict_ahead=1, sampling=False, n_sample=100):
    df_alarm = df_alarm.loc[df_alarm.description.isin(alarms_list)]
    df_alarm = df_alarm.loc[:, ['dt_start_down', 'description']]
    df_alarm.sort_values('dt_start_down', inplace=True)
    df_alarm.reset_index(drop=True, inplace=True)
    df_alarm.loc[:, ['date_1']] = df_alarm.dt_start_down - pd.Timedelta(hours=predict_ahead+hist_data_length)
    df_alarm.loc[:, ['date_2']] = df_alarm.dt_start_down - pd.Timedelta(hours=predict_ahead)
    df_alarm.loc[:, ['date_1_to_date_2']] = df_alarm['date_1'].astype('str') + ' - ' + df_alarm['date_2'].astype('str')
    if sampling:
        df_alarm = df_alarm.sample(n=n_sample, random_state=42)
    return df_alarm

def select_data_with_no_alarm(df_alarm, df_mc, hist_data_length=3, predict_ahead=1, sampling=False, n_sample=100):
    alarm_date_time = list(df_alarm['dt_start_down'].values)
    df_no_alarm = df_mc.loc[~df_mc['date_time'].isin(alarm_date_time)]
    df_no_alarm = df_no_alarm.loc[:, ['date_time']]
    df_no_alarm.loc[:, ['description']] = 'No Alarm'
    df_no_alarm.loc[:, ['date_1']] = df_no_alarm.date_time - pd.Timedelta(hours=predict_ahead+hist_data_length)
    df_no_alarm.loc[:, ['date_2']] = df_no_alarm.date_time - pd.Timedelta(hours=predict_ahead)
    df_no_alarm.loc[:, ['date_1_to_date_2']] = df_no_alarm['date_1'].astype('str') + ' - ' + df_no_alarm['date_2'].astype('str')
    df_no_alarm.rename(columns = {'date_time':'dt_start_down'}, inplace = True)
    if sampling:
        df_no_alarm = df_no_alarm.sample(n=n_sample, random_state=42)
        df_no_alarm.reset_index(drop=True, inplace=True)
    return df_no_alarm

def join_data_with_and_without_alarm(df_with_alarm, df_without_alarm):
    df_with_and_without_alarm = pd.concat([df_with_alarm, df_without_alarm])
    df_with_and_without_alarm.sort_values('dt_start_down', inplace=True)
    df_with_and_without_alarm.reset_index(drop=True, inplace=True)
    return df_with_and_without_alarm

def map_alarm_data_to_mc(df_mc, df_alarm_preprocessed, filter_data=False, x=30):
    mapped = df_alarm_preprocessed.loc[df_alarm_preprocessed.index.repeat((df_alarm_preprocessed['date_2'] - df_alarm_preprocessed['date_1']).dt.total_seconds())]
    mapped['Date'] = mapped['date_1'] + pd.to_timedelta(mapped.groupby(level=0).cumcount(), unit='s')
    mapped = mapped.reset_index(drop=True)
    mapped = pd.merge(df_mc, mapped, left_on='date_time', right_on='Date')
    # select (alarm) data with more than x (mc data) timestep
    if filter_data:
        filter_ = mapped.groupby('date_1_to_date_2').size()[mapped.groupby('date_1_to_date_2').size() > x].index
        mapped = mapped.loc[mapped.date_1_to_date_2.isin(filter_)]
    mapped.drop(['date_1', 'date_2', 'Date'], axis=1, inplace=True)
    mapped.set_index('date_1_to_date_2', inplace=True)
    return mapped

def feature_extraction(features, mapped_alarm_to_machine):
    mapped = mapped_alarm_to_machine.loc[:, features + ['description']]
    mapped_features = mapped.iloc[:, :-1]
    mapped_label = pd.DataFrame(mapped.iloc[:, -1])
    mapped_label.reset_index(inplace=True)
    mapped_label.drop_duplicates('date_1_to_date_2', inplace=True)
    mapped_label.set_index('date_1_to_date_2', inplace=True)
    mapped_extracted_features = pd.DataFrame()
    for i in range(len(features)):
        mapped_extracted_features['mean.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].mean()
        mapped_extracted_features['median.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].quantile(0.5)
        mapped_extracted_features['std.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].std()
        mapped_extracted_features['qntl1.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].quantile(0.25)
        mapped_extracted_features['qntl3.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].quantile(0.75)
        mapped_extracted_features['min.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].min()
        mapped_extracted_features['max.' + features[i]] = mapped_features.groupby('date_1_to_date_2')[features[i]].max()
    mapped_extracted_features = mapped_extracted_features.sort_index()
    mapped_label = mapped_label.sort_index()
    extracted = pd.merge(mapped_extracted_features, mapped_label, left_index=True, right_index=True)
    extracted.iloc[:, :-1] = extracted.iloc[:, :-1].astype('float32')
    return extracted

def binary_label_encoder(dataset):
    dataset.loc[:, ['binary_description']] = dataset['description'].apply(lambda x: 'No Alarm' if x == 'No Alarm' else 'Alarm')
    dataset.loc[:, ['binary_description_encoded']] = dataset['binary_description'].apply(lambda x: 0 if x == 'No Alarm' else 1)
    return dataset

def train_test_spliter(features_extracted_data, test_size=0.2):
    train_set, test_set = sklearn.model_selection.train_test_split(features_extracted_data, test_size=test_size, random_state=42, stratify=features_extracted_data.iloc[:,-1].values)
    return train_set, test_set

def multiclass_label_encoder(dataset):
    other_class = [
                'Blower problem**                                  ',
                'Processing cell/Piping problem**                  ',
                'Processing pump problem**                         ',
                'Strip jam**                                       ',
                'Phase check error**                               ',
                'Facility Supply Faulty                            '
                ]
    dataset.loc[:, ['description_5_classes']] = dataset['description']
    mask = dataset['description'].isin(other_class)
    dataset.loc[mask, 'description_5_classes'] = 'Other'
    dataset.loc[:, ['description_encoded']] = dataset['description_5_classes'].astype('category').cat.codes
    return dataset

def transform(train_set, test_set):
    X = train_set.iloc[:,:-3].values.astype('float32')
    y = train_set.iloc[:,-1].values.astype(int)
    X_test = test_set.iloc[:,:-3].values.astype('float32')
    y_test = test_set.iloc[:,-1].values.astype(int)
    return X, y, X_test, y_test

def binary_alarm_preprocess(df_alarm, df_mc):
    
    alarms_list = [
                'Others Machine Problem**                          ',
                'UNLOAD PROB***',
                'INLOAD PROB***',
                'Rectifiers problem**                              ',
                'Blower problem**                                  ',
                'Processing cell/Piping problem**                  ',
                'Processing pump problem**                         ',
                'Strip jam**                                       ',
                'Phase check error**                               ',
                'Facility Supply Faulty                            ',
                ]

    main_features = [
                'Converyer_Belt_Speed_m_min', 'Blower_Pressure_Bar', 'MatteTIn_Curent_Amp'
                ]

    df_alarm_meco_04, df_mc_meco_04 = select_type_mc(df_alarm, df_mc)
    df_with_alarm = select_data_with_alarm(df_alarm_meco_04, alarms_list, sampling=True, n_sample=56)
    df_no_alarm = select_data_with_no_alarm(df_alarm, df_mc_meco_04, sampling=True, n_sample=50)
    df_with_and_without_alarm = join_data_with_and_without_alarm(df_with_alarm, df_no_alarm)
    mapped = map_alarm_data_to_mc(df_mc_meco_04, df_with_and_without_alarm)
    extracted_features = feature_extraction(main_features, mapped)
    dataset_with_encoded_label = binary_label_encoder(extracted_features)
    train_set, test_set = train_test_spliter(dataset_with_encoded_label)
    X, y, X_test, y_test = transform(train_set, test_set)
    return X, y, X_test, y_test

def multiclass_alarm_preprocess(df_alarm, df_mc):
    
    alarms_list = [
                'Others Machine Problem**                          ',
                'UNLOAD PROB***',
                'INLOAD PROB***',
                'Rectifiers problem**                              ',
                'Blower problem**                                  ',
                'Processing cell/Piping problem**                  ',
                'Processing pump problem**                         ',
                'Strip jam**                                       ',
                'Phase check error**                               ',
                'Facility Supply Faulty                            ',
                ]

    main_features = [
                'Converyer_Belt_Speed_m_min', 'Blower_Pressure_Bar', 'MatteTIn_Curent_Amp'
                ]
    
    df_alarm_meco_04, df_mc_meco_04 = select_type_mc(df_alarm, df_mc)
    df_with_alarm = select_data_with_alarm(df_alarm_meco_04, alarms_list)
    mapped = map_alarm_data_to_mc(df_mc_meco_04, df_with_alarm)
    extracted_features = feature_extraction(main_features, mapped)
    dataset_with_encoded_label = multiclass_label_encoder(extracted_features)
    train_set, test_set = train_test_spliter(dataset_with_encoded_label)
    X, y, X_test, y_test = transform(train_set, test_set)
    return X, y, X_test, y_test