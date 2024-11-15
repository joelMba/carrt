#!/usr/bin/env python
# coding: utf-8

"""
CARRT (Classification based on Association Rules Ranking with Topsis Method) 
is an associative classification framework using the 'TopKClassRules' algorithm for association rule mining. 
It applies multicriteria analysis, particularly the TOPSIS method, for ranking association rules.
"""

import pandas as pd
import Postraitement
import Prediction
from spmf import Spmf

class CARRT:
    """
    Usage:
    Import the 'carrt' module, then create an instance of the 'CARRT' class after completing preprocessing and data preparation.
    Inputs must use the Spmf format.
    """

    def __init__(self, k, min_confidence, consequent_item_ids, number_of_rules_per_class):
        """
        Initialize CARRT with parameters:
        - k: Number of association rules to discover (positive integer).
        - min_confidence: Minimum confidence for association rules (value in [0,1]).
        - consequent_item_ids: Target class item(s) allowed in the rule consequent.
        - number_of_rules_per_class: Number of rules per class for classifier construction.
        """
        self.k = k
        self.min_confidence = min_confidence
        self.consequent_item_ids = consequent_item_ids
        self.number_of_rules_per_class = number_of_rules_per_class
        self.classifier = pd.DataFrame()

    def fit(self, data_train, data_train_structured_version):
        """
        Constructs the classifier from training data.
        Inputs:
        - data_train: Path to training data in spmf format (.txt).
        - data_train_structured_version: Structured version of training data (dataframe) for calculating additional rule quality metrics.
        
        Outputs:
        - self.classifier: DataFrame with the best rules for each class.
        - self.train_result: Summary of training data results.
        
        Note: The target class should be explicitly named "Class".
        """
        input_path = data_train
        output_path = "rules.txt"

        # Run the TopKClassRules algorithm
        top_k_class_rules = Spmf("TopKClassRules", input_filename=input_path, output_filename=output_path, arguments=[self.k, self.min_confidence, self.consequent_item_ids])
        top_k_class_rules.run()

        # Process and refine rules
        rules = Postraitement.deleteRedundantRules(output_path)
        rules = Postraitement.lif_conv(data_train_structured_version, rules)
        df_rules = Postraitement.classement_with_conv(rules)

        # Build classifier with the best rules
        df_rules_result = Prediction.classifier2(data_train_structured_version, df_rules, self.number_of_rules_per_class)
        self.classifier = df_rules_result[0]
        self.train_result = df_rules_result[1]

    def predict(self, test_data):
        """
        Predicts classes for test data.
        Input:
        - test_data: DataFrame containing the test data.
        
        Output:
        - DataFrame with predicted values for each instance in the test set.
        """
        return Prediction.prediction3(test_data, self.classifier)

    def evaluation(self, test_data, dataset_name):
        """
        Evaluates the model on test data.
        Inputs:
        - test_data: DataFrame with test data.
        - dataset_name: Name of the dataset.
        
        Outputs:
        - Evaluation metrics: accuracy, precision, recall, f1-score.
        - Confusion matrix.
        """
        return Prediction.experiment(test_data, self.classifier, self.number_of_rules_per_class, dataset_name)
