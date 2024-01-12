#!/usr/bin/env python
# coding: utf-8

# In[147]:

"""
CARRT (Classification based on Association Rules Ranking with Topsis Method), is an associative classification framework that employs the 'TopKClassRules' algorithm for association rule mining. 
Additionally, it incorporates multicriteria analysis, particularly the TOPSIS method, for ranking association rules.

"""


import pandas as pd
import Postraitement
import Prediction
from spmf import Spmf

class CARRT(object):
    """
    To use this framework, you need to start by import "carrt" module and creating an instance of the 'CARRT' class after completing the preprocessing and data preparation steps.
    Inputs datas used Spmf format 
    """
        
    def __init__(self, k, min_confidence, consequent_item_ids, number_of_rules_per_class):
        """
        Description of inputs parameters:
        --------------------------
        k : representing the number of association rules to be discovered (a positive integer)
        min_confidence : representing the minimum confidence that the association rules should have (a value in [0,1] representing a percentage)
        consequent_item_ids : Item that allowed to appear in the consequent of rules (target class) 
        number_of_rules_per_class : Number of rules per class to choose for classifier construction.
        """
        self.k = k
        self.min_confidence = min_confidence
        self.consequent_item_ids = consequent_item_ids
        self.classifier = pd.DataFrame()
        self.number_of_rules_per_class = number_of_rules_per_class
        
    def fit(self, data_train, data_train_structuredVersion):
        """
        This function allows the construction of the classifier from the training data.
        Inputs:
        - data_train: path to the training data in spmf format (.txt)
        - data_train_structuredVersion: The structured version of the training data (in dataframes) will allow the potential calculation of additional association rule quality measures
        (such as lift, conviction, etc.)
        Outputs: A list where the first element of the list is the classifier, and the second element of the list contains a summary of the results on the training data
        - self.classifier: dataframe containing the resulting classifier (the "number_of_rules_per_class" best rules leading to each class)
        - self.train_result: summary of the results on the training data
                 
        Note: It is recommended to explicitly name the target class as "Class.".
        """
    
        # Define the input and output file paths
        input_path = data_train
        output_path = "rules.txt"

        # Run the algorithm on the train data
        top_k_class_rules = Spmf("TopKClassRules", input_filename=input_path, output_filename=output_path, arguments=[self.k, self.min_confidence,self.consequent_item_ids])

        top_k_class_rules.run()
        
        #Post-traitement des règles d'associations
        rules = Postraitement.deleteRedundantRules(output_path)
        
        #Calcul de mesure de règles d'associations supplémentaires
        rules = Postraitement.lif_conv(data_train_structuredVersion, rules)
        df_rules = Postraitement.classement_with_conv(rules)

        df_rules_10 = Prediction.classifier2(data_train_structuredVersion, df_rules, self.number_of_rules_per_class)
        
        self.classifier = df_rules_10[0]
        
        self.train_result = df_rules_10[1]
        
    def predict(self, test_data):
        """
        Input: Path to the test data (dataframe)
        Outputs: DataFrame containing the predicted values for each individual in the test set
        """
        test_data = test_data
        df_predict = Prediction.prediction3(test_data, self.classifier)
        return df_predict
    
    def evaluation(self, test_data, dataset_name):
        """
        Enables experimentation on test data.
        Inputs:
        - test_data: Path to the test data (dataframe)
        - dataset_name: Name of the dataset (text)
        Outputs:
        - Accuracy and precision curves based on the number of rules in the classifier (ranging from 1 to "number_of_rules_per_class")
        - Table containing accuracy, precision, recall, f1...
        - Confusion matrix
        """
        test_data = test_data
        dataset_name = dataset_name
        df_evaluation = Prediction.experiment(test_data, self.classifier, self.number_of_rules_per_class, dataset_name)
        return df_evaluation






