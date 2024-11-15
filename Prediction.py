#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics

def prediction3(df, df_rules):
    """
    Perform prediction using rule-based inference on the input DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input data for prediction. Must include the 'Class' column.
        df_rules (pd.DataFrame): Rule set for inference.

    Returns:
        pd.DataFrame: DataFrame with added predictions.
    """
    # Identify the most frequent value in the 'Class' column
    max_occurrence_value = df['Class'].value_counts().idxmax()
    
    # Extract and temporarily remove 'Class' column
    code_clients = df['Class'].tolist()
    df = df.drop(['Class'], axis=1)
    
    # Perform rule matching and prediction for each row
    df_predictions = []
    for i in range(len(df)):
        intervention = dict(df.iloc[i])
        matching_rules = matching_Rule3(intervention, df_rules)
        predicted_class = categorie2(matching_rules, intervention, max_occurrence_value)
        df_predictions.append(predicted_class)

    # Restore 'Class' column and add predictions
    df['Class'] = code_clients
    df['prediction'] = pd.DataFrame(df_predictions)['Classe_predite'].tolist()

    # Handle cases with no matching rules
    df = no_matching_rule(df, df_rules)
    return df

def matching_Rule3(intervention, df_rules):
    """
    Match rules based on the attributes of a given intervention.
    
    Parameters:
        intervention (dict): Dictionary of intervention attributes.
        df_rules (pd.DataFrame): Rule set to match.

    Returns:
        pd.DataFrame: DataFrame of matching rules.
    """
    df_rules = df_rules.filter(['Antecedent', 'Conclusion', 'Support', 'Confiance', 'NbItems', 'score', 'rank'], axis=1)
    df_rules['Classe'] = df_rules['Conclusion'].str.split('=').str[1]
    df_result = pd.DataFrame(columns=df_rules.columns)

    # Check for exact matches
    for i, rule in df_rules.iterrows():
        antecedent = antecedent_to_dict(rule['Antecedent'])
        if dict_subsetOf_dict(intervention, antecedent):
            df_result = pd.concat([df_result, pd.DataFrame([rule])])

    # Check for partial matches if no exact matches
    if df_result.empty:
        for i, rule in df_rules.iterrows():
            antecedent = antecedent_to_dict(rule['Antecedent'])
            common_keys = set(intervention.keys()).intersection(antecedent.keys())
            common_values = [intervention[key] == antecedent[key] for key in common_keys]
            if sum(common_values) >= 2:
                df_result = pd.concat([df_result, pd.DataFrame([rule])])

    return df_result.drop_duplicates()

def antecedent_to_dict(antecedent):
    """
    Convert an antecedent string into a dictionary.
    
    Parameters:
        antecedent (str): Antecedent string in 'key=value' format.

    Returns:
        dict: Parsed antecedent as a dictionary.
    """
    items = re.split(' ', antecedent.strip())
    return {item.split('=')[0]: item.split('=')[1] for item in items}

def dict_subsetOf_dict(dict1, dict2):
    """
    Check if dict2 is a subset of dict1.

    Parameters:
        dict1 (dict): Base dictionary.
        dict2 (dict): Subset to check.

    Returns:
        bool: True if dict2 is a subset of dict1, False otherwise.
    """
    return all(dict1.get(key) == val for key, val in dict2.items())

def categorie2(df_rule_c, intervention, max_occurrence_value):
    """
    Determine predicted class based on candidate rules.

    Parameters:
        df_rule_c (pd.DataFrame): Candidate rules.
        intervention (dict): Intervention attributes.
        max_occurrence_value: Default class in case of ambiguity.

    Returns:
        pd.DataFrame: Intervention with predicted class.
    """
    modal_classes = df_rule_c['Classe'].mode()

    if len(modal_classes) == 1:
        intervention['Classe_predite'] = modal_classes[0]
    else:
        top_scores = df_rule_c[df_rule_c['Classe'].isin(modal_classes)].groupby('Classe')['score'].mean()
        best_class = top_scores.idxmax()
        intervention['Classe_predite'] = best_class

    return pd.DataFrame([intervention])

def no_matching_rule(df, df_rules):
    """
    Assign random class to observations with no matching rules.

    Parameters:
        df (pd.DataFrame): Data with predictions.
        df_rules (pd.DataFrame): Rule set for reference.

    Returns:
        pd.DataFrame: Data with predictions filled for unmatched cases.
    """
    classes = df_rules['Conclusion'].str.split('=').str[1].unique()
    for i, row in df.iterrows():
        if pd.isna(row['prediction']):
            df.at[i, 'prediction'] = random.choice(classes)
    return df

# Metrics calculation functions
def accuracy(y_test, y_pred):
    return metrics.accuracy_score(y_test, y_pred)

def precision(y_test, y_pred):
    return metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)

def recall(y_test, y_pred):
    return metrics.recall_score(y_test, y_pred, average='macro', zero_division=0)

def f1(y_test, y_pred):
    return metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)

def print_confusion_matrix(y_test, y_pred):
    """
    Display confusion matrix as a heatmap.
    """
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(pd.DataFrame(cm), annot=True, fmt='d', cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
