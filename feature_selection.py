from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from typing import List, Tuple
from collections import Counter

import numpy as np
import os
import csv

selection_ratio = 0.75
total_score = 100
p = 2  # Pangkat untuk distribusi

def power_distribution(features: List[str]) -> List[Tuple[str, float]]:
    n = len(features)
    
    # Hitung bobot mentah untuk setiap peringkat
    weights = [(n - i) ** p for i in range(n)]
    total_weight = sum(weights)
    
    # Hitung skor final dengan normalisasi ke total_score
    scores = [(features[i], (weights[i] / total_weight) * total_score) for i in range(n)]
    
    return scores

def intersection(chi2_top, anova_top, info_gain_top):
    intersection_features = chi2_top & anova_top & info_gain_top
    print("\n=========Intersection Features: ")
    print(intersection_features)
    return intersection_features

def select_weighted_top_features(chi2_scores, anova_scores, info_gain_scores, selection_length, dataset):
    feature_scores = {}

    for feature, chi2_val in chi2_scores:
        anova_val = dict(anova_scores).get(feature, 0)
        info_val = dict(info_gain_scores).get(feature, 0)
        
        valid_scores = [v for v in [chi2_val, anova_val, info_val] if not np.isnan(v)]
        avg_score = np.mean(valid_scores) if valid_scores else 0
        
        feature_scores[feature] = avg_score

    weighted_ranking = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    weighted_columns = [f[0] for f in weighted_ranking]
    weighted_top = {f[0] for f in weighted_ranking[:selection_length]}

    exportLogFeatureSelection(dataset, "weighted", weighted_ranking)

    print("\n=========Weighted")
    print(weighted_ranking)
    print(weighted_columns)
    print(weighted_top)
    return weighted_columns

def feature_voting(intersection_set, average_set, selection_length, dataset):
    all_features = list(intersection_set) + list(average_set)
    feature_counts = Counter(all_features)

    voting_ranking = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    voting_columns = [f[0] for f in voting_ranking[:selection_length]]

    exportLogFeatureSelection(dataset, "voting", voting_ranking)
    
    print("\n=========Voting Best Features:")
    print(voting_ranking)
    print(voting_columns)

    return voting_ranking, voting_columns

def withChi2(X, y, selection_length, dataset):
    chi_scores, _ = chi2(X, y)
    chi2_ranking = sorted(zip(X.columns, chi_scores), key=lambda x: x[1], reverse=True)
    chi2_columns = [f[0] for f in chi2_ranking]
    chi2_top = {f[0] for f in chi2_ranking[:selection_length]}
    chi2_weighted = power_distribution(chi2_columns)

    exportLogFeatureSelection(dataset, "chi2", chi2_ranking)
    exportLogFeatureSelection(dataset, "weighted_chi2", chi2_weighted)

    print("\n===Ranking Fitur berdasarkan Chi-Square:")
    print(chi2_ranking)
    print(chi2_columns)
    print(chi2_top)
    print(chi2_weighted)
    return chi2_weighted, chi2_top

def withAnova(X, y, selection_length, dataset):
    anova_scores, _ = f_classif(X, y)
    anova_ranking = sorted(zip(X.columns, anova_scores), key=lambda x: x[1], reverse=True)
    anova_columns = [f[0] for f in anova_ranking]
    anova_top = {f[0] for f in anova_ranking[:selection_length]}
    anova_weighted = power_distribution(anova_columns)

    exportLogFeatureSelection(dataset, "anova", anova_ranking)
    exportLogFeatureSelection(dataset, "weighted_anova", anova_weighted)

    print("\n===Ranking Fitur berdasarkan ANOVA F-test:")
    print(anova_ranking)
    print(anova_columns)
    print(anova_top)
    print(anova_weighted)
    return anova_weighted, anova_top

def withInfo_gain(X, y, selection_length, dataset):
    info_gain_scores = mutual_info_classif(X, y)
    info_gain_ranking = sorted(zip(X.columns, info_gain_scores), key=lambda x: x[1], reverse=True)
    info_gain_columns = [f[0] for f in info_gain_ranking]
    info_gain_top = {f[0] for f in info_gain_ranking[:selection_length]}
    info_gain_weighted = power_distribution(info_gain_columns)

    exportLogFeatureSelection(dataset, "info_gain", info_gain_ranking)
    exportLogFeatureSelection(dataset, "weighted_info_gain", info_gain_weighted)

    print("\n===Ranking Fitur berdasarkan Information Gain:")
    print(info_gain_ranking)
    print(info_gain_columns)
    print(info_gain_top)
    print(info_gain_weighted)
    return info_gain_weighted, info_gain_top

def checkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

def exportLogFeatureSelection(dataset, name, feature):
    OUT_DIR = os.getenv('OUT_DIR')
    dataset = dataset.replace("/","-")
    dataset = dataset.replace(":","")
    dataset = dataset.replace(".binetflow","/")
    folder = OUT_DIR+dataset
    checkDir(folder)
    filename = folder+name+"_fs_log.csv"

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(feature)