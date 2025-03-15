"""Botnet Detection with Hybrid Feature Selection using Rank Weighted Analysis"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import os
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import feature_selection as featureSelection
import pre_processing as preprocessing
import classification as classification

from dotenv import load_dotenv
from datetime import datetime

def loopAllInDataset(selectedAlgorithm, dirName):
    file_names = os.listdir(dirName)
    for f in file_names:
        print("====RUN data: ",dirName+f)
        mainModule(selected_algorithm,dirName+f)

def mainModule(selected_algorithm, file):
    now = datetime.now()
    
    OUT_DIR = os.getenv('OUT_DIR')

    print("===============| Load Data |=================")
    data = pd.read_csv(file)
    data['Label'] = data['Label'].apply(preprocessing.label)
    data  = preprocessing.main(pd, data)
    print(data.isnull().sum())

    exclude_features = ['StartTime', 'Label']
    data_excluded = data[exclude_features]
    data_features = data.drop(columns=exclude_features)

    print("===============| Normalization |=================")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_features)
    data_scaled = pd.DataFrame(data_scaled, columns=data_features.columns, index=data.index)
    data_final = pd.concat([data_scaled, data_excluded], axis=1)

    target_column = 'Label'
    X = data_final.drop(columns=[target_column])
    y = data_final[target_column]

    print("===============| Feature Selection |=================")
    selection_length = int(len(X.columns) * featureSelection.selection_ratio)
    print("Select best: ",selection_length)

    chi2_weighted, chi2_top = featureSelection.withChi2(X, y, selection_length, file)
    anova_weighted, anova_top = featureSelection.withAnova(X, y, selection_length, file)
    info_gain_weighted, info_gain_top = featureSelection.withInfo_gain(X, y, selection_length, file)
    intersection_features = featureSelection.intersection(chi2_top, anova_top, info_gain_top)
    weighted_columns = featureSelection.select_weighted_top_features(chi2_weighted, anova_weighted, info_gain_weighted, selection_length, file)
    voting_ranking, voting_columns = featureSelection.feature_voting(intersection_features, weighted_columns, selection_length, file)

    print("===============| Splitting |=================")
    train, test = preprocessing.split_by_class(data_final)
    X_train = train[voting_columns]
    y_train = train[target_column]

    X_test = test[voting_columns]
    y_test = test[target_column]
    
    model = classification.train(X_train, y_train, selected_algorithm)
    tp, tn, fp, fn, accuracy, precision, recall, f1 = classification.test(model, X_test, y_test)
    
    dict = {
        "CreatedAt": now,
        "Algorithm": selected_algorithm,
        "ClassificationContext": "Hybrid Feature Selection with Intersection, weight and voting",
        "SelectedFeatures": str(voting_columns),
        "File": file,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }

    logFilePath = OUT_DIR+str(int(now.timestamp()))+'_log.txt'
    outputFilePath = OUT_DIR+'classification_results.csv'
    file_exists = os.path.exists(outputFilePath) and os.path.getsize(outputFilePath) > 0
    field_names = ['CreatedAt', 'Algorithm', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ClassificationContext', 'SelectedFeatures', 'File']
    with open(outputFilePath, 'a', newline='') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
        
        if not file_exists:
            dict_object.writeheader()

        dict_object.writerow(dict)


if __name__ == "__main__":
    load_dotenv()
    CTU_DIR = os.getenv('CTU_DIR')
    NCC_DIR = os.getenv('NCC_DIR')
    NCC_2_DIR = os.getenv('NCC_2_DIR')

    selected_algorithm = classification.menu()
    
    loopAllInDataset(selected_algorithm, CTU_DIR)
    loopAllInDataset(selected_algorithm, NCC_DIR)
    loopAllInDataset(selected_algorithm, NCC_2_DIR)
