from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", help = "Please input the file name which should be in same folder")
    args = parser.parse_args()
    return args

def run_experiments(df):
    output_column = "outcome"
    cols = [i for i in df.columns if i != output_column]
    print("X columns are :" ,cols)
    print("Y column is :" ,f"'{output_column}'")
    y = df[output_column]
    X = df[cols]
    start = time.time()
    model = XGBClassifier()
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    end = time.time()
    return {"accuracy": scores.mean(), "time": end-start }

def main():
    args = get_args()
    df = pd.read_csv(args.file_name)
    print(run_experiments(df))

if __name__ == '__main__':
    main()

# python xg_boost_python.py --file-name sample_100.csv
# python xg_boost_python.py --file-name sample_1000.csv
# python xg_boost_python.py --file-name sample_10000.csv
# python xg_boost_python.py --file-name sample_100000.csv
# python xg_boost_python.py --file-name sample_1000000.csv
# python xg_boost_python.py --file-name sample_10000000.csv