import pandas as pd
import sklearn

def select_type_mc(df_defect, df_mc, mc_type='MECO-04'):
    df_defect_mc_type = df_defect.loc[df_defect.MACHINE == mc_type]
    df_defect_mc_type['DATETIME_OUT'] =  pd.to_datetime(df_defect_mc_type['DATETIME_OUT'])
    df_defect_mc_type.sort_values('DATETIME_OUT', inplace=True)
    df_defect_mc_type.reset_index(drop=True, inplace=True)
    df_mc_mc_type = df_mc.copy()
    df_mc_mc_type['date_time'] =  pd.to_datetime(df_mc_mc_type['date_time'])
    df_mc_mc_type.sort_values('date_time', inplace=True)
    df_mc_mc_type.reset_index(drop=True, inplace=True)
    return df_defect_mc_type, df_mc_mc_type

def clean_target_col(df_defect):
    df_defect_new = df_defect.copy()
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('COPPER RESIDUE : คราบเศษทองแดง)', 'COPPER RESIDUE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('PLATING CONTRAST : ความแตกต่างของเนื้อเพลท', 'PLATING CONTRAST')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('NON PLATE/INCOMPLETE PLATING : ลีดแดง / เพลทไม่ติด )', 'NON PLATE/INCOMPLETE PLATING')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('INCOMPLETE  MOLD : โมลด์ไม่เต็ม / คอมปาวด์ติดหน้าโมลด์ )', 'INCOMPLETE  MOLD')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('ILLEGIBLE MARK : มาร์คเลือน )', 'ILLEGIBLE MARK')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('FANNED LEAD : แฟนลีด', 'FANNED LEAD')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('EXCESSIVE SOLDER : ตะกั่วยื่น)', 'EXCESSIVE SOLDER')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('PLATE BURR : เนื้อเพลทเป็นเสี้ยน )', 'PLATE BURR')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('EXPOSED GLUE ON TERMINAL SIDE : เศษกาวโผล่ด้านเทอร์มินอล)', 'EXPOSED GLUE ON TERMINAL SIDE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('VOID : รูบนผิวแพคเกจ', 'VOID')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('PLATING SCRATCH : รอยขีดข่วนบริเวณผิวของแพคเกจ )', 'PLATING SCRATCH')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('RESIDUE WIRE ON TERMINAL SIDE : เศษลวดโผล่ด้านเทอร์มินอล', 'RESIDUE WIRE ON TERMINAL SIDE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('DAMAGE STRIP FROM M/C : สตริปเสียจากเครื่อง )', 'DAMAGE STRIP FROM M/C')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('INTERNAL VOID : โพรงอากาศภายในแพคเกจ ดูจาก X - ray )', 'INTERNAL VOID : X - ray')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('SCRATCH ON PANEL : รอยขีดข่วนบนพาเนล )', 'SCRATCH ON PANEL')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('PLATING BUBBLE : เนื้อเพลทเป็นฟองอากาศ )', 'PLATING BUBBLE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('3/O ESCAPEES : เติดออฟเกิน)', '3/O ESCAPEES')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('SOLDER TEST : โซลเดอร์เทสต์)', 'SOLDER TEST')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('DISCOLOR  LEAD : ลีดเปลี่ยนสี )', 'DISCOLOR  LEAD')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('SOLDER NODULE : ลีดเป็นเม็ด )', 'SOLDER NODULE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('DAMAGED STRIP FROM M/C : สตริปเสียจากเครื่อง', 'DAMAGED STRIP FROM M/C')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CRACK PACKAGE : แพคเกจร้าว', 'CRACK PACKAGE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('WATER STAIN : คราบน้ำ', 'WATER STAIN')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('RELIABILITY TEST : ทดสอบความทนทานของยูนิต )', 'RELIABILITY TEST')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('STRIP JAM / DAMAGE STRIP : สตริปเสียจากเครื่อง )', 'STRIP JAM / DAMAGE STRIP')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('EXPOSED WIRE : ลวดโผล่ )', 'EXPOSED WIRE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CONTAM ON PANEL : สิ่งสกปรกติดบนกพาแนล )', 'CONTAM ON PANEL')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('EXTERNAL VOID/PIN HOLE : รูบนผิวแพคเกจ )', 'EXTERNAL VOID/PIN HOLE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CONTAM BODY AND LEAD : หมึกเปื้อนแพคเกจและขาลีด )', 'CONTAM BODY AND LEAD')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('DAMAGED TERMINAL/ Dap : เทอร์มินอล/แดบเกิดความเสียหาย)', 'DAMAGED TERMINAL/ Dap')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('FADE MARK : สิ่งสกปรกติดในร่องมาร์ค )', 'FADE MARK')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('DAMAGED / BROKEN WIRE : ลวดเสียรูป / ลวดขาด )', 'DAMAGED / BROKEN WIRE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('LEADFRAME DEFECT : เฟรมเสียหาย', 'LEADFRAME DEFECT')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('STAIN ON LEAD : คราบสกปรกบนขาลีด )', 'STAIN ON LEAD')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('INCOMPLETE BUFF : รอยขัดไม่สม่ำเสมอ', 'INCOMPLETE BUFF')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('UNEVEN/NO SPACING : มาร์คช่องไฟไม่ตรง )', 'UNEVEN/NO SPACING')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CRACK / SEPARATION PANEL : พาเนลร้าว / เป็นรอยแยก )', 'CRACK / SEPARATION PANEL')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('SOLDER PEELING : ตะกั่วลอก )', 'SOLDER PEELING')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CRACK / SEPARATION : แพคเกจร้าว / เป็นรอยแยก )', 'CRACK / SEPARATION')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('EXPOSED FOREIGN MATERIAL (DOT) : เศษโลหะฝังในคอมปาวด์ด้านเทอร์มินอล (จุด)', 'EXPOSED FOREIGN MATERIAL (DOT)')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('EXPOSED FOREIGN MATERIAL (DOT) : ?????????????????????????????????? (???)', 'EXPOSED FOREIGN MATERIAL (DOT)')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('FLASH / RESIN BLEED : แฟลชดำ / เหลือง )', 'FLASH / RESIN BLEED')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('FLASH / RESIN BLEED : ?????? / ?????? )', 'FLASH / RESIN BLEED')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('NON PLATE : ลีดแดง)', 'NON PLATE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('NON PLATE : ??????)', 'NON PLATE')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CONTAM ON PACKAGE : สิ่งสกปรกติดบนแพคเกจ )', 'CONTAM ON PACKAGE_1')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CONTAM ON PACKAGE : สิ่งสกปรกติดบนแพคเกจ', 'CONTAM ON PACKAGE_2')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CONTAM ON TERMINAL OR DAP : สิ่งสกปรกติดบนเทอร์มินอลหรือแดพ', 'CONTAM ON TERMINAL OR DAP_1')
    df_defect_new.loc[:, ['REJECT_DESCRIPTION']] = df_defect_new.loc[:, ['REJECT_DESCRIPTION']].replace('CONTAM ON TERMINAL OR DAP : สิ่งสกปรกติดบนเทอร์มินอลหรือแดป )', 'CONTAM ON TERMINAL OR DAP_2')
    df_defect_new.update(df_defect_new.loc[df_defect_new['REJECT_CODE']=='FF'].loc[:, ['REJECT_DESCRIPTION']].replace('SCRATCH PACKAGE : รอยขีดข่วนบนแพคเกจ )', 'SCRATCH PACKAGE_1'))
    df_defect_new.update(df_defect_new.loc[df_defect_new['REJECT_CODE']=='PI'].loc[:, ['REJECT_DESCRIPTION']].replace('SCRATCH PACKAGE : รอยขีดข่วนบนแพคเกจ )', 'SCRATCH PACKAGE_2'))
    df_defect_new.loc[:, ['REJECT_CODE_DESCRIPTION']] = df_defect_new['REJECT_DESCRIPTION'] + '_' + df_defect_new['REJECT_CODE']
    df_defect_new.sort_values('DATETIME_OUT', inplace=True)
    df_defect_new.reset_index(drop=True, inplace=True)
    return df_defect_new

# select data with different REJECT_CODE (REJECT_CODE_DESCRIPTION) for every DATETIME_OUT
def select_data_with_diff_reject_code(df_defect):
    df_defect_new = df_defect.copy()
    distinct_datetime_out = list(df_defect_new['DATETIME_OUT'].value_counts()[df_defect_new['DATETIME_OUT'].value_counts() == 1].index)
    df_defect_new = df_defect_new.loc[df_defect_new.DATETIME_OUT.isin(distinct_datetime_out)]
    df_defect_new.sort_values('DATETIME_OUT', inplace=True)
    df_defect_new.reset_index(drop=True, inplace=True)
    return df_defect_new

def select_data_with_defect(df_defect, hist_data_length=3, predict_ahead=1, sampling=False, n_sample=100):
    df_defect = df_defect.loc[~df_defect.REJECT_CODE_DESCRIPTION.isnull()]
    df_defect = df_defect.loc[:, ['DATETIME_OUT', 'REJECT_CODE_DESCRIPTION']]
    df_defect.sort_values('DATETIME_OUT', inplace=True)
    df_defect.reset_index(drop=True, inplace=True)
    df_defect.loc[:, ['date_1']] = df_defect.DATETIME_OUT - pd.Timedelta(hours=predict_ahead+hist_data_length)
    df_defect.loc[:, ['date_2']] = df_defect.DATETIME_OUT - pd.Timedelta(hours=predict_ahead)
    df_defect.loc[:, ['date_1_to_date_2']] = df_defect['date_1'].astype('str') + ' - ' + df_defect['date_2'].astype('str')
    if sampling:
        df_defect = df_defect.sample(n=n_sample, random_state=42)
    return df_defect

def select_data_with_no_defect(df_defect, hist_data_length=3, predict_ahead=1, sampling=False, n_sample=100):
    df_no_defect = df_defect.loc[df_defect.REJECT_CODE_DESCRIPTION.isnull()]
    df_no_defect = df_no_defect.loc[:, ['DATETIME_OUT', 'REJECT_CODE_DESCRIPTION']]
    df_no_defect.sort_values('DATETIME_OUT', inplace=True)
    df_no_defect.reset_index(drop=True, inplace=True)
    df_no_defect.loc[:, ['date_1']] = df_no_defect.DATETIME_OUT - pd.Timedelta(hours=predict_ahead+hist_data_length)
    df_no_defect.loc[:, ['date_2']] = df_no_defect.DATETIME_OUT - pd.Timedelta(hours=predict_ahead)
    df_no_defect.loc[:, ['date_1_to_date_2']] = df_no_defect['date_1'].astype('str') + ' - ' + df_no_defect['date_2'].astype('str')
    df_no_defect.loc[:, ['REJECT_CODE_DESCRIPTION']] = 'No Defect'
    if sampling:
        df_no_defect = df_no_defect.sample(n=n_sample, random_state=42)
        df_no_defect.reset_index(drop=True, inplace=True)
    return df_no_defect

def join_data_with_and_without_defect(df_with_defect, df_without_defect):
    df_with_and_without_defect = pd.concat([df_with_defect, df_without_defect])
    df_with_and_without_defect.sort_values('DATETIME_OUT', inplace=True)
    df_with_and_without_defect.reset_index(drop=True, inplace=True)
    return df_with_and_without_defect

def map_defect_data_to_mc(df_mc, df_defect_preprocessed, filter_data=False, x=30):
    mapped = df_defect_preprocessed.loc[df_defect_preprocessed.index.repeat((df_defect_preprocessed['date_2'] - df_defect_preprocessed['date_1']).dt.total_seconds())]
    mapped['Date'] = mapped['date_1'] + pd.to_timedelta(mapped.groupby(level=0).cumcount(), unit='s')
    mapped = mapped.reset_index(drop=True)
    mapped = pd.merge(df_mc, mapped, left_on='date_time', right_on='Date')
    # select (defect) data with more than x (mc data) timestep
    if filter_data:
        filter_ = mapped.groupby('date_1_to_date_2').size()[mapped.groupby('date_1_to_date_2').size() > x].index
        mapped = mapped.loc[mapped.date_1_to_date_2.isin(filter_)]
    mapped.drop(['date_1', 'date_2', 'Date'], axis=1, inplace=True)
    mapped.set_index('date_1_to_date_2', inplace=True)
    return mapped

def feature_extraction(features, mapped_defect_to_machine):
    mapped = mapped_defect_to_machine.loc[:, features + ['REJECT_CODE_DESCRIPTION']]
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
    dataset.loc[:, ['binary_REJECT_CODE_DESCRIPTION']] = dataset['REJECT_CODE_DESCRIPTION'].apply(lambda x: 'No Defect' if x == 'No Defect' else 'Defect')
    dataset.loc[:, ['binary_REJECT_CODE_DESCRIPTION_encoded']] = dataset['binary_REJECT_CODE_DESCRIPTION'].apply(lambda x: 0 if x == 'No Defect' else 1)
    return dataset

def train_test_spliter(features_extracted_data, test_size=0.2):
    train_set, test_set = sklearn.model_selection.train_test_split(features_extracted_data, test_size=test_size, random_state=42, stratify=features_extracted_data.iloc[:,-1].values)
    return train_set, test_set

def multiclass_label_encoder(dataset):
    other_class = list(dataset['REJECT_CODE_DESCRIPTION'].value_counts()[3:].index)
    dataset.loc[:, ['REJECT_CODE_DESCRIPTION_4_classes']] = dataset['REJECT_CODE_DESCRIPTION']
    mask = dataset['REJECT_CODE_DESCRIPTION'].isin(other_class)
    dataset.loc[mask, 'REJECT_CODE_DESCRIPTION_4_classes'] = 'Other'
    dataset.loc[:, ['REJECT_CODE_DESCRIPTION_encoded']] = dataset['REJECT_CODE_DESCRIPTION_4_classes'].astype('category').cat.codes
    return dataset

def transform(train_set, test_set):
    X = train_set.iloc[:,:-3].values.astype('float32')
    y = train_set.iloc[:,-1].values.astype(int)
    X_test = test_set.iloc[:,:-3].values.astype('float32')
    y_test = test_set.iloc[:,-1].values.astype(int)
    return X, y, X_test, y_test

def binary_defect_preprocess(df_defect, df_mc):
    
    main_features = [
        'Converyer_Belt_Speed_m_min', 'Blower_Pressure_Bar', 'MatteTIn_Curent_Amp'
            ]
    
    df_defect_meco_04, df_mc_meco_04 = select_type_mc(df_defect, df_mc, mc_type='MECO-04')
    df_defect_meco_04_new = clean_target_col(df_defect_meco_04)
    df_defect_meco_04_diff_reject_code = select_data_with_diff_reject_code(df_defect_meco_04_new)
    df_with_defect = select_data_with_defect(df_defect_meco_04_diff_reject_code, sampling=True, n_sample=1100)
    df_without_defect = select_data_with_no_defect(df_defect_meco_04_diff_reject_code, sampling=True, n_sample=1063)
    df_with_and_without_defect = join_data_with_and_without_defect(df_with_defect, df_without_defect)
    mapped = map_defect_data_to_mc(df_mc_meco_04, df_with_and_without_defect, filter_data=True, x=30)
    extracted_features = feature_extraction(main_features, mapped)
    dataset_with_encoded_label = binary_label_encoder(extracted_features)
    train_set, test_set = train_test_spliter(dataset_with_encoded_label, test_size=0.1999578148)
    X, y, X_test, y_test = transform(train_set, test_set)
    return X, y, X_test, y_test

def multiclass_defect_preprocess(df_defect, df_mc):
    
    main_features = [
        'Converyer_Belt_Speed_m_min', 'Blower_Pressure_Bar', 'MatteTIn_Curent_Amp'
            ]
    
    df_defect_meco_04, df_mc_meco_04 = select_type_mc(df_defect, df_mc, mc_type='MECO-04')
    df_defect_meco_04_new = clean_target_col(df_defect_meco_04)
    df_defect_meco_04_diff_reject_code = select_data_with_diff_reject_code(df_defect_meco_04_new)
    df_with_defect = select_data_with_defect(df_defect_meco_04_diff_reject_code)
    mapped = map_defect_data_to_mc(df_mc_meco_04, df_with_defect, filter_data=True, x=30)
    extracted_features = feature_extraction(main_features, mapped)
    dataset_with_encoded_label = multiclass_label_encoder(extracted_features)
    train_set, test_set = train_test_spliter(dataset_with_encoded_label, test_size=0.1999578148)
    X, y, X_test, y_test = transform(train_set, test_set)
    return X, y, X_test, y_test