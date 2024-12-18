#!/usr/bin/env python
# coding: utf-8

"""
This module implements post-processing functions for association rules generated by the 
TopKClassAssociationRules algorithm. It includes functions to remove conflicting or redundant rules, 
compute additional metrics like lift and conviction, and rank rules using the TOPSIS method. 
This module also contains the implementation of the oversampling method using SMOTEN method 
"""

import pandas as pd
import numpy as np
import re
from scikitmcda.topsis import TOPSIS
from scikitmcda.constants import MAX, MIN, LinearMinMax_, LinearSum_

from imblearn.over_sampling import SMOTEN
from collections import Counter
import pandas as pd

# Remove conflicting rules
def delete_conflicting_rules(rules):
    """
    Removes conflicting rules, i.e., rules with the same antecedent but different conclusions.
    
    Parameters:
    - rules: DataFrame containing the association rules.
    
    Returns:
    - filtered_rules: DataFrame without conflicting rules.
    """
    groups = rules.groupby('Antecedent')['Conclusion'].nunique()
    antecedents_to_remove = groups[groups > 1].index.tolist()
    filtered_rules = rules[~rules['Antecedent'].isin(antecedents_to_remove)]
    return filtered_rules

# Remove redundant rules
def delete_redundant_rules(input_file):
    """
    Removes redundant and conflicting rules from a text file containing extracted rules.
    
    Parameters:
    - input_file: Path to the file containing the rules (format: .txt).
    
    Returns:
    - df: DataFrame with non-redundant and non-conflicting rules.
    """
    column_names = ["Antecedent", "Conclusion", "Support", "Confidence"]
    df = pd.DataFrame(columns=column_names)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    rules = []
    for line in lines:
        line = line.strip()
        components = re.split(r' ==> | #SUP: | #CONF: |\n', line)
        rules.append({"Antecedent": components[0], "Conclusion": components[1], "Support": float(components[2]), "Confidence": float(components[3])})
    df = pd.DataFrame(rules)

    # Remove conflicting rules
    df = delete_conflicting_rules(df)

    return df

# Convert antecedents to a dictionary
def antecedent_to_dict(antecedent):
    """
    Converts an antecedent string to a dictionary of key-value pairs.
    
    Parameters:
    - antecedent: String representing the antecedent.
    
    Returns:
    - result: Dictionary of key-value pairs.
    """
    items = re.split(r' ', antecedent.strip())
    return {item.split('=')[0]: item.split('=')[1] for item in items}

# Check if a dictionary is a subset of another
def is_dict_subset(dict1, dict2):
    """
    Checks if dict2 is a subset of dict1.
    
    Parameters:
    - dict1: Main dictionary.
    - dict2: Dictionary to check.
    
    Returns:
    - bool: True if dict2 is a subset of dict1, otherwise False.
    """
    return all(dict1.get(key) == val for key, val in dict2.items())

# Calculate the support of antecedents
def antecedent_support(antecedent, data):
    """
    Calculates the support of a given antecedent in a dataset.
    
    Parameters:
    - antecedent: String representing the antecedent.
    - data: DataFrame containing the dataset.
    
    Returns:
    - support: Support of the antecedent in the dataset.
    """
    dict_antecedent = antecedent_to_dict(antecedent)
    return sum(is_dict_subset(row.to_dict(), dict_antecedent) for _, row in data.iterrows())

# Compute lift and conviction metrics
def calculate_lift_conviction(train_data, rules_df):
    """
    Computes lift and conviction metrics for the association rules.
    
    Parameters:
    - train_data: DataFrame of training data.
    - rules_df: DataFrame of association rules.
    
    Returns:
    - rules_df: DataFrame with lift and conviction columns added.
    """
    rules_df['Class'] = rules_df['Conclusion'].apply(lambda x: x.split('=')[1])
    class_counts = train_data['Class'].value_counts().to_dict()

    # Calculate supports and metrics
    rules_df['Consequent_support'] = rules_df['Class'].map(class_counts) / len(train_data)
    rules_df['Support'] = rules_df['Support'] / len(train_data)
    rules_df['Lift'] = rules_df['Confidence'] / rules_df['Consequent_support']
    rules_df['Conviction'] = (1 - rules_df['Consequent_support']) / (1 - rules_df['Confidence'])

    # Handle infinite and NaN values in conviction
    rules_df['Conviction'].replace([np.inf, -np.inf], np.nan, inplace=True)
    max_conviction = rules_df['Conviction'].max(skipna=True) + 1
    rules_df['Conviction'].fillna(max_conviction, inplace=True)

    return rules_df[["Antecedent", "Conclusion", "Support", "Confidence", "Lift", "Conviction"]]

# Visualize association rules
def visualize_association_rules(df_rules, number_of_rules):
    """
    Plots a scatter plot of the rules based on support and confidence.
    
    Parameters:
    - df_rules: DataFrame containing the rules.
    - number_of_rules: Number of rules to visualize.
    """
    df_rules.head(number_of_rules).plot.scatter(x='Confidence', y='Support', colormap='viridis', title=f"Scatter plot of {number_of_rules} rules")

# Rank rules using TOPSIS with entropy-based weights
def rank_rules_with_topsis(df):
    """
    Ranks association rules using the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method.
    
    Steps:
    1. Adds an index column and calculates the number of items in each antecedent.
    2. Defines the evaluation criteria for the rules.
    3. Uses the TOPSIS method to rank the rules:
       - Entropy method is applied to calculate weights for each criterion based on their variability.
       - Criteria are normalized and signals (maximize/minimize) are set for each.
       - The ranking is determined by the TOPSIS decision process.
    
    Parameters:
    - df: DataFrame containing the association rules to be ranked.
      Required columns: 'Support', 'Confidence', 'Lift', 'Conviction', 'Antecedent'.
    
    Returns:
    - df: DataFrame with two additional columns:
        - 'Score': The TOPSIS performance score for each rule.
        - 'Rank': The rank of each rule based on the performance score.
    """
    # Assign a unique number to each rule for identification
    df['Number'] = range(1, len(df) + 1)
    
    # Calculate the number of items in the antecedent
    df['NbItems'] = df['Antecedent'].apply(lambda x: len(x.strip().split(" ")))
    
    # Define the criteria for evaluation
    criteria = ["Support", "Confidence", "Lift", "Conviction", "NbItems"]
    
    # Initialize TOPSIS
    topsis = TOPSIS()
    
    # Provide the criteria data and identifiers to TOPSIS
    topsis.dataframe(df[criteria].values, df['Number'].tolist(), criteria)
    
    # Specify whether to maximize or minimize each criterion
    # MAX for criteria where higher values are better, MIN where lower values are better
    topsis.set_signals([MAX, MAX, MAX, MAX, MIN])
    
    # Apply entropy method to calculate weights for the criteria
    # The entropy method assigns higher weights to criteria with greater variability
    topsis.set_weights_by_entropy(normalization_method_for_entropy=LinearSum_)
    
    # Perform the TOPSIS decision-making process using linear min-max normalization
    topsis.decide(LinearMinMax_)
    
    # Add the TOPSIS results to the DataFrame
    df['Score'] = topsis.df_decision['performance score']
    df['Rank'] = topsis.df_decision['rank']
    
    return df


def apply_smoten(df, class_column, target_ratio=0.2, random_state=42, k_neighbors=5):
    """
    Applies SMOTEN to handle imbalanced data by oversampling minority classes.
    
    Parameters:
        df (pd.DataFrame): The input dataset.
        class_column (str): The name of the column containing class labels.
        target_ratio (float): The ratio (proportion of the majority class count) to determine
                              the target increase for minority classes.
        random_state (int): Random state for reproducibility.
        k_neighbors (int): Number of nearest neighbors for SMOTEN.
    
    Returns:
        pd.DataFrame: The resampled dataset with balanced classes.
    """
    # Separate features and target
    X = df.drop(class_column, axis=1)
    y = df[class_column]

    # Get the count of each class
    class_counts = dict(Counter(y))

    # Identify the majority class
    majority_class = max(class_counts, key=class_counts.get)

    # Calculate the target gap
    target_gap = int(target_ratio * class_counts[majority_class])

    # Build the sampling_strategy dictionary
    sampling_strategy = {}
    for class_label, count in class_counts.items():
        if class_label != majority_class:
            target_size = count + target_gap
            sampling_strategy[class_label] = target_size

    # Initialize SMOTEN
    smote_nc = SMOTEN(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)

    # Fit and resample the dataset
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)

    # Combine resampled features and target into a DataFrame
    df_resampled = pd.DataFrame(data=X_resampled, columns=X.columns)
    df_resampled[class_column] = y_resampled

    return df_resampled


