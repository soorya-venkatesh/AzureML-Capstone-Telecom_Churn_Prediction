from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
#filepath= "https://drive.google.com/uc?id=1837VRDmRvDa43w_XnzTdY9bofWWg4k-P&export=download"
filepath="https://raw.githubusercontent.com/soorya-venkatesh/nd00333-capstone/master/starter_file/Telco_customer_churn.csv"
ds = TabularDatasetFactory.from_delimited_files(path = filepath)


def clean_data(data):
    
    df = data.to_pandas_dataframe()
    #Total charges has a datatype object eventhough it is continious in nature. So we are converting it to numeric type
    df['Total Charges']=pd.to_numeric(df['Total Charges'], errors='coerce')
    #replacing null values with median
    df['Total Charges'].fillna(df['Total Charges'].median(),inplace=True)
    #We create a new column called entertainment combining 2 pre existing columns stream movies and stream TV
    df.loc[(df['Streaming Movies'] == 'Yes')&(df['Streaming TV'] == 'Yes'),'Entertainment'] = 2
    df.loc[(df['Streaming Movies'] == 'No internet service')&(df['Streaming TV'] == 'No internet service'),'Entertainment'] = 0
    df['Entertainment'].fillna(1,inplace=True)
    #Removing columns which are not going to be used for modelling purpose
    df.drop(['CustomerID' ,'Count' ,'Country' ,'State' ,'Lat Long' ,'Latitude' ,'Longitude','Zip Code','Streaming TV', 'Streaming Movies','Churn Label','Churn Score', 'CLTV', 'Churn Reason'], axis=1, inplace=True)
    #label encoding categorical data
    label_encoded_columns=['Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Contract','Payment Method','Entertainment',
    'Gender','Senior Citizen','Partner','Dependents','Phone Service','Paperless Billing']
    le=LabelEncoder()
    for i in label_encoded_columns:
        df[i]=le.fit_transform(df[i])
    #Frequency encoding column 'city' since it has a high cardinality
    fe=df.groupby('City').size()/len(df)
    df.loc[:,'City_freq_encode']=df['City'].map(fe)

    X = df.drop(['Churn Value','City'],axis=1)
    y = df['Churn Value']  
    
    return X,y


#df= pd.read_csv('Telco_customer_churn.csv')
x,y =clean_data(ds)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y)
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=1.0, help="maximum depth per tree")
    parser.add_argument('--n_estimators', type=int, default=100, help="Total decision tress that can be used")
    parser.add_argument('--min_samples_split', type=int, default=100, help="The minimum number of samples required to split an internal node")
    parser.add_argument('--min_samples_leaf', type=int, default=100, help="The minimum number of samples required to be at a leaf node")

    args = parser.parse_args()

    run.log("Max Depth:", np.int(args.max_depth))
    run.log("Total base estimators", np.int(args.n_estimators))
    run.log("Minimum samples per split", np.int(args.min_samples_split))
    run.log("Min samples leaf", np.int(args.min_samples_leaf))

    model = RandomForestClassifier(class_weight="balanced",max_depth=args.max_depth,min_samples_split=args.min_samples_split,min_samples_leaf=args.min_samples_leaf,n_estimators=args.n_estimators).fit(x_train,y_train)

    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    os.makedirs('outputs', exist_ok=True)
    
    joblib.dump(model, 'outputs/model.joblib')
    run.log("AUC-ROC", np.float(auc))


run = Run.get_context()
if __name__ == '__main__':
    main()
