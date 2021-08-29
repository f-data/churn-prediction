import numpy as np
import pandas as pd
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path,'Churn Modeling.csv')
    df = pd.read_csv(train_file_path)
    return df
    
def process_data(df):
     return  (df.assign(IsMale = lambda x: np.where(df.Gender== "Male", 1 ,0))
              .pipe(pd.get_dummies, columns=['Geography'])
              .drop(['Gender'], axis=1)
             )
             
    
    
def write_data(df):
    proccessed_data_path =os.path.join(os.path.pardir,'data','processed')
    write_train_path = os.path.join(proccessed_data_path,'train.csv')
    write_test_path = os.path.join(proccessed_data_path,'train.csv')
    df.to_csv(write_train_path)
    df.to_csv(write_test_path)
    
    
if __name__ == '__main__':
    df= read_data()
    df= process_data(df)
    write_data(df) 
    df.info()
