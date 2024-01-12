#!/usr/bin/env python
# coding: utf-8



    
def prediction3(df,df_rules):
    """- Cette fonction prends en entrée une dataframe comportant les données à prédire
    - La variable de prédiction 'Class' est retirée avant le processus de prédiction, puis rajoutée à fin de la fonction
    - En sortie on a donc la dataframe constituée des colonnes descriptive, la colonne prédictive (état avant et après la prédiction)
    - Le processus de prédiction s'éffectue en 2 phases: Etant donnée une intervention/opération, on recherche l'ensemble des règles
    candidates, ensuite on applique un algorithme de sélection de la meilleure règle"""
    import pandas as pd
    df_rules = df_rules
    
    # count the occurrences of each unique value in the column
    value_counts = df['Class'].value_counts()

    # find the value that occurs the most
    max_occurrence_value = value_counts.idxmax()
    
    codeCliens = df['Class'].tolist()
    df = df.drop(['Class'], axis=1)
    intervention = dict(df.iloc[0])
    df1 = matching_Rule3(intervention,df_rules)


    df2 = categorie2 (df1,intervention, max_occurrence_value)
    for i in range(1,len(df)):
        intervention = dict(df.iloc[i])
        df1 = matching_Rule3(intervention,df_rules)
        df2 = pd.concat([df2,categorie2(df1,intervention,max_occurrence_value)])
    df['Class'] = codeCliens
    df['prediction'] = df2['Classe_predite'].tolist()
    # Le cas ou on a des observations pour lesquelles on a pas de règles qui matchent
    df = no_matching_rule(df,df_rules)
    return df
    



def matching_Rule3(intervention, df_rules):
    """ 
    - Une règle candidate ici est une règle dont règle dont toute la prémisse est contenu dans les attributs de l'intervention
    - Dans le cas ou on a pas de règle dont toute la prémisse est contenue dans l'intervention, on sélectionne les règles dont au moins un attribut match
    """
    import pandas as pd
    #df_rules = pd.read_excel('Donnees/Paca2/Rules_Base.xlsx')
    for key in intervention:
        intervention[key] = str(intervention[key])
    df_rules = df_rules.filter(['Antecedent','Conclusion','Support','Confiance','NbItems','score','rank'],axis=1)
    df_rules['Classe'] = [df_rules['Conclusion'].iloc[i].split('=')[1] for i in range(len(df_rules))]
    column_names = df_rules.columns.tolist()
    df_result = pd.DataFrame(columns = column_names)

    for i in range(len(df_rules)):
        antecedent = antecedent_to_dict(df_rules['Antecedent'].iloc[i])
        if dict_subsetOf_dict(intervention,antecedent)=="True":
            #df_result = df_result.append(df_rules.iloc[i])
            df_result.loc[len(df_result)] = df_rules.iloc[i]
            #df_result = pd.concat([df_result, df_rules.iloc[i]], ignore_index=True)
    
    if (len(df_result)==0):
        #check if at least two key-value pairs from the first dictionary are present in the second dictionary
        for i in range(len(df_rules)):
            antecedent = antecedent_to_dict(df_rules['Antecedent'].iloc[i])
            common_keys = set(intervention.keys()).intersection(antecedent.keys())
            if len(common_keys) >= 2:
                common_values = [intervention[key] == antecedent[key] for key in common_keys]
                true_count = common_values.count(True)
                if true_count >= 2:
                    #df_result = df_result.append(df_rules.iloc[i])
                    df_result.loc[len(df_result)] = df_rules.iloc[i]
                    #df_result = pd.concat([df_result, df_rules.iloc[i]], ignore_index=True)
    
    df_result = df_result.drop_duplicates()
    return df_result




def antecedent_to_dict(antecedent):
    """Cette fonction prends en entrée un dictionnaire(qui représente la prélisse d'une règle) et la transforme
    en un dictionnaire clé-valeur, qui est retourné en sortie"""
    result=dict()
    import re
    items = re.split(' ',str(antecedent.rstrip()))
    for i in range(len(items)):
        item = re.split('=',str(items[i]))
        result[item[0]] = item[1]
    return result


# In[9]:


def dict_subsetOf_dict(dict1,dict2):
    """ Cette fonction sert à tester si le dictionnaire 'dict2' est un sous-ensemble de ductionnaire 'dict1'
    Retourne en sortie une valeure booléenne"""
    # Using all() + items()
    # Check if one dictionary is subset of other
    result = all(dict1.get(key, None) == val for key, val in dict2.items())
    return str(result)



def categorie2 (df_rule_c,intervention,max_occurrence_value):
    """ Cette fonction prends en entrée 2 paramètres: une dataframe qui représente les règles candidates, 
    et une intervention (qui est un dictionnaire)
    - On recherche ensuite la classe modale. Si elle est unique, alors on considère que sa valeur représente la classe prédite
    - Par contre si on a plusieurs valeurs de classe modales, on applique un algorithme qui permet de rechercher la classe eyant le plus
    grand score (parmis les classe modales uniquement)"""
    import pandas as pd
    import random
    dic = dict(df_rule_c['Classe'].mode())

    if len(dic)==1 :
        intervention['Classe_predite']=dic[0] 
    if len(dic)>1:
        df_rule_c1 = df_rule_c[df_rule_c['Classe'].isin(list(dic.values()))]
        df_rule_c1.reset_index(drop=True, inplace=True)# To reset index
        
        #Prendre la classe qui a le score moyen le plus élevé
        df_rule_c2 = df_rule_c1.groupby("Classe")["score"].mean().reset_index(name="Score_moyen")    
        
        index_rule_elct = df_rule_c2['Score_moyen'].idxmax()
        
        intervention['Classe_predite'] = df_rule_c2['Classe'].iloc[index_rule_elct]
        
    df = pd.DataFrame(intervention,index=[0])
    
    return df

    
def no_matching_rule(df,df_rules):
    """ Cette fonction permet de prendre en considération le cas ou on a aucune règle qui matche avec le cas à prédire. 
    Dans ce cas on retourne une classe aléatoire parmis l'ensemble des classes du classifieur """
    import pandas as pd
    import random
    # Split the 'Original_Column' and create a new column
    df_rules['Class'] = df_rules['Conclusion'].str.split('=').str[1]
    classes = df_rules['Class'].unique().tolist()
    
    for i in range(len(df)):
        if pd.isnull(df['prediction'].iloc[i]): 
            #codeClients.remove(df['codeClient'].iloc[i])
            df['prediction'].iloc[i] = random.choice(classes)
            #classes.append(df['Class'].iloc[i])
    return df


# In[8]:


def accuracy(y_test, y_pred):
    """ Cette fonction prends en entrée 2 listes comportant respectivement les vérités et les valeurs prédites et 
    retourne en sortie la précision (accuracy)"""
    from sklearn import metrics
    #y_test = [str(x) for x in y_test]
    #y_pred = [str(x) for x in y_pred]
    return metrics.accuracy_score(y_test, y_pred)
    
def precision(y_test, y_pred):
    """ Cette fonction prends en entrée 2 listes comportant respectivement les vérités et les valeurs prédites et 
    retourne en sortie la précision (accuracy)"""
    from sklearn import metrics
    #y_test = [str(x) for x in y_test]
    #y_pred = [str(x) for x in y_pred]
    return metrics.precision_score(y_true=y_test, y_pred=y_pred,average='macro', zero_division=0)
    
def recall(y_test, y_pred):
    """ Cette fonction prends en entrée 2 listes comportant respectivement les vérités et les valeurs prédites et 
    retourne en sortie la précision (accuracy)"""
    from sklearn import metrics
    #y_test = [str(x) for x in y_test]
    #y_pred = [str(x) for x in y_pred]
    return metrics.recall_score(y_true=y_test, y_pred=y_pred,average='macro', zero_division=0)
    
def f1(y_test, y_pred):
    """ Cette fonction prends en entrée 2 listes comportant respectivement les vérités et les valeurs prédites et 
    retourne en sortie la précision (accuracy)"""
    from sklearn import metrics
    #y_test = [str(x) for x in y_test]
    #y_pred = [str(x) for x in y_pred]
    return metrics.f1_score(y_true=y_test, y_pred=y_pred,average='macro', zero_division=0)


# In[9]:


def printConfusionMatrix (y_test, y_pred):
    """ Cette fonction prends en entrée 2 listes comportant respectivement les vérités et les valeurs prédites et 
    retourne en sortie la matrice de confusion"""
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.tree import export_graphviz 
    from IPython.display import Image  
    import pydotplus
    import pandas as pd
    
    # Create a new figure for the confusion matrix plot
    plt.figure()
    cm = metrics.confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
    cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
    plt.title('Confusion matrix')
    
def accuracy_precision_curves(df_results, dataset_name):
        import matplotlib.pyplot as plt
        # Create a new figure for the confusion matrix plot
        plt.figure()
        plt.plot(df_results['Nb_Class_Rule'], df_results['accuracy'], marker='o', color='black', linestyle='dotted', label='Accuracy')
    
        # Plot precision curve with black solid line
        plt.plot(df_results['Nb_Class_Rule'], df_results['precision'], marker='x', color='black', linestyle='-', label='Precision')
    
        plt.title(dataset_name)
        plt.xlabel("Nombre de règles par classes (k)")
        plt.ylabel("Score (%)")
        plt.grid(True)
        plt.legend()
        plt.show()
        #return plt2


# In[6]:


    
def experiment(df_test, df_rules, k, dataset_name):

    import pandas as pd
    dataset_name = dataset_name
    
    df_rules = df_rules.filter(['Antecedent','Conclusion','Support','Confiance','NbItems','score','rank'],axis=1)
    columns_result = ['Nb_Class_Rule']
    df_result=pd.DataFrame(columns = columns_result)
    classes = df_rules['Conclusion'].unique().tolist()
    column_names = df_rules.columns.tolist()
    dict_result = dict()
    dict_result_2 = dict()
    dict_result_3 = dict()
    dict_result_4 = dict()
    dict_result_5 = dict()
    for i in range(1,k+1):       
        dfi= pd.DataFrame(columns = column_names)  
        Nb_Of_Rule = 0
        for classe in classes:
            df_classe = df_rules[df_rules['Conclusion']==classe]
            df_classe = df_classe.sort_values(by='score',ascending=False, ignore_index=True)
            df_classe = df_classe.head(i)
            Nb_Of_Rule = Nb_Of_Rule + len(df_classe)
            dfi = pd.concat([dfi,df_classe])
        dfi = prediction3(df_test,dfi)
        dict_result[i] = accuracy(df_test.Class,dfi.prediction)
        dict_result_2[i] = precision(df_test.Class,dfi.prediction)
        dict_result_3[i] = recall(df_test.Class,dfi.prediction)
        dict_result_4[i] = f1(df_test.Class,dfi.prediction)
        dict_result_5[i] = Nb_Of_Rule
    df_result['Nb_Class_Rule']=list(dict_result.keys())
    df_result['Nb_Of_Rule']=list(dict_result_5.values())
    df_result['accuracy']=list(dict_result.values())
    df_result['precision']=list(dict_result_2.values())
    df_result['recall']=list(dict_result_3.values())
    df_result['f1']=list(dict_result_4.values())
    
    accuracy_precision_curves(df_result, dataset_name)
    
    printConfusionMatrix (df_test.Class, dfi.prediction)
    
    
    
    
    return df_result
   

# In[ ]:
    
    
def classifier(df_rules,k):
    import pandas as pd
    df_rules = df_rules.filter(['Antecedent','Conclusion','Support','Confiance','Lift','Conviction','Leverage','NbItems','score','rank'],axis=1)
    columns_result = ['Nb_Class_Rule']
    df_result=pd.DataFrame(columns = columns_result)
    classes = df_rules['Conclusion'].unique().tolist()
    column_names = df_rules.columns.tolist()
    for i in range(1,k+1):       
        dfi= pd.DataFrame(columns = column_names)  
        Nb_Of_Rule = 0
        for classe in classes:
            df_classe = df_rules[df_rules['Conclusion']==classe]
            df_classe = df_classe.sort_values(by='score',ascending=False, ignore_index=True)
            df_classe = df_classe.head(i)
            Nb_Of_Rule = Nb_Of_Rule + len(df_classe)
            dfi = pd.concat([dfi,df_classe])
    return dfi
    



def remove_kth_best_rule(df_rules, k):
    import pandas as pd
    # Filtrage des colonnes pertinentes
    df_rules = df_rules.filter(['Antecedent','Conclusion','Support','Confiance','Lift','Leverage','NbItems','score','rank'], axis=1)
    
    # Initialisation d'une liste pour stocker les DataFrames résultants
    result_dfs = []
    
    # liste des classes uniques
    classes = df_rules['Conclusion'].unique()
    
    # Parcours de chaque classe
    for classe in classes:
        # Filtrage des règles pour la classe actuelle
        class_rules = df_rules[df_rules['Conclusion'] == classe]
        
        # Trie des règles par score en ordre décroissant
        class_rules = class_rules.sort_values(by='score', ascending=False, ignore_index=True)
        
        # Vérification si k est valide pour cette classe
        if k < len(class_rules):
            # Supprimer la k-ième meilleure règle
            class_rules = class_rules.drop([k-1], axis=0)
        
        # Ajout du DataFrame résultant à la liste
        result_dfs.append(class_rules)
    
    # Concaténation de tous les DataFrames de résultat
    result_df = pd.concat(result_dfs, ignore_index=True)
    
    return result_df
   
   
def classifier2(train_data, rules, threshold):
    rules = rules
    i = 1
    df_rules = classifier(rules, i)
    df_predict = prediction3(train_data, df_rules)
    acc = accuracy(train_data.Class, df_predict.prediction)
    train_acc = dict() 
    train_acc[i] = acc
    while i <= threshold :
        #df_rules = classifier(rules, i)
        df_predict = prediction3(train_data, classifier(rules, i))
        current_acc = accuracy(train_data.Class, df_predict.prediction)
        if current_acc < acc :
            #classifier ajustment
            rules = remove_kth_best_rule(rules, i)
        else :
            df_rules = classifier(rules, i)
            acc = current_acc
            train_acc[i] = acc
            i = i + 1
    return [df_rules,train_acc]
    