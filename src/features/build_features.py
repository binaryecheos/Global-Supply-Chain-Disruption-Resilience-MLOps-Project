import pandas as pd
import numpy as np
import joblib

import os

def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")
    
def save_data(df: pd.DataFrame, file_path: str):
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")



def drop_features(df):
    try:
        df = df.drop(['Order_ID','Order_Date'], axis=1)
        df['Disruption_Event'] = df['Disruption_Event'].fillna('No Disruption')
        return df
    except Exception as e:
            raise Exception(f"Error dropping features: {e}")


def encoding_features(df):
    try:
        encoded_df = pd.get_dummies(df,columns=['Route_Type','Destination_City','Origin_City','Delivery_Status','Transportation_Mode','Product_Category','Disruption_Event'])
        return encoded_df
    except Exception as e:
        raise Exception(f"Error encoding features: {e}")
    

def main():
    raw_data_path = r"data/raw"
    processsed_data_path = r"data/processed"

    train_data = load_data(os.path.join(raw_data_path, "train.csv"))
    test_data = load_data(os.path.join(raw_data_path, "test.csv"))
    train_cleaned_data = drop_features(train_data)
    test_cleaned_data = drop_features(test_data)
    train_processed_data = encoding_features(train_cleaned_data)
    test_processed_data = encoding_features(test_cleaned_data)

    columns = train_processed_data.drop(['Mitigation_Action_Taken'], axis=1).columns

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(columns, os.path.join(models_dir, "columns.pkl"))

    os.makedirs(processsed_data_path, exist_ok=True)
    save_data(train_processed_data, os.path.join(processsed_data_path, "train_processed.csv"))
    save_data(test_processed_data, os.path.join(processsed_data_path, "test_processed.csv"))

if __name__ == "__main__":
    main()

