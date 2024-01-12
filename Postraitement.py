#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" Ce module implémente les fonctions de post-traitements de règles d'associations en sortie de l'algorithme TopKClassAssociationRules,
en vue de la construction du classifieur. Particulièrement la fonction de ranking des règles d'associations qui est basée sur la méthode Topsis"""



def deleteConflictualRules (rules):
    """
    Cette fonction permet de supprimer les règles conflictuelles (2 règles sont dites conflictuelles lorsqu'elles ont les memes antécedants avec des conclusions differentes.
    Entrée: dataframe contenant la liste des règles d'associations.
    """
    # Regrouper par antécédent et vérifier les conclusions uniques
    groups = rules.groupby('Antecedent')['Conclusion'].nunique()

    # Obtenir les antécédents ayant plusieurs conclusions uniques
    antecedents_to_remove = groups[groups > 1].index.tolist()

    # Supprimer les enregistrements avec des antécédents ayant plusieurs conclusions differentes
    filtered_rules = rules[~rules['Antecedent'].isin(antecedents_to_remove)]

    return filtered_rules


# In[4]:


def deleteRedundantRules(input_file):
    """Cette fonction prends entrée le chemin d'accès au fichier txt des règles extraites et supprime les doublons
    - Une règle est considérée comme redondante si on peut trouver dans notre base de règles une autre règle 
    de meme support/confiance, avec des sous-ensembles d'items de l'une incluses dans l'autre
    - Cette fonction fait également appel à la fonction de suppression des règles conflictuelles
    - En sortie on a donc une dataframe qui contient uniquement les règles non redondantes, 
    qui est également exporté dans un fichier excel"""
    
    import pandas as pd
    import re
    column_names = ["Antecedent","Conclusion", "Support", "Confiance"]
    df = pd.DataFrame(columns = column_names)
    with open(input_file,'r') as f:
        lines = f.readlines()
    dicts =[]
    for line in lines:
        line = line.strip()
        #line = re.split(' #SUP: | #CONF: |\n', line)
        #rule = {"Regle":line[0], "Support":line[1], "Confiance":line[2]}
        line = re.split(' ==> | #SUP: | #CONF: |\n', line)
        rule = {"Antecedent":line[0],"Conclusion":line[1], "Support":line[2], "Confiance":line[3]}
        dicts.append(rule)
    #print(dicts)
    df = df.append(dicts, ignore_index=True, sort=False)
    
    # Suppression des règles conflictuelles
    df = deleteConflictualRules (df)

    return df



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

def dict_subsetOf_dict(dict1,dict2):
    """ Cette fonction sert à tester si le dictionnaire 'dict2' est un sous-ensemble de ductionnaire 'dict1'
    Retourne en sortie une valeure booléenne"""
    # Using all() + items()
    # Check if one dictionary is subset of other
    result = all(dict1.get(key, None) == val for key, val in dict2.items())
    return str(result)    
    
#fonction à mettre dans utils.py
def antecedant_support(antecedent, data):
#import du module de prédiction (provisoire en attendant la mise en place du module utils.py)
    #import Prediction
    dict_antecedent = antecedent_to_dict(antecedent)
    support = 0
    for i in range(len(data)) :
        dict_row = data.iloc[i].to_dict()
        if dict_subsetOf_dict(dict_row,dict_antecedent) == "True" :
            support = support + 1
    return support  
    

def lif_conv (train_data,df_rules):
    """ 
    Cette fonction permet de calculer le Lift et la conviction des règles d'associations, le support et la confiance étant connus d'avance.
    Entrées: 2 dataframes en entrées
      - train_data: dataframe contenant les données d'entrainement
      - df_rules: dataframe contenant les règles d'associations. 
    Sortie: Dtataframe contenant pour chacune des règles d'association, le lift et la conviction associés (en plus des autres mesures)
    """
    import numpy as np
    df_rules['Class'] = df_rules['Conclusion'].apply(lambda x: x.split('=')[1])  
    Class_count = train_data['Class'].value_counts().to_dict()
    df_rules['Consequent_support'] = df_rules['Class'].map(Class_count)
    train_data = train_data.astype(str, errors='ignore')
    df_rules['Support'] = df_rules['Support'].astype(float)
    df_rules['Confiance'] = df_rules['Confiance'].astype(float)
    df_rules['Consequent_support'] = df_rules['Consequent_support'].astype(float)
    df_rules['Support'] = df_rules['Support'] / len(train_data)
    df_rules['Consequent_support'] = df_rules['Consequent_support'] / len(train_data)
    
    #lift
    df_rules['Lift'] = df_rules['Confiance'] / df_rules['Consequent_support']
    #conviction
    df_rules['Conviction'] = (1 - df_rules['Consequent_support']) / (1 - df_rules['Confiance'])
    # Remplacement des valeurs "Inf" par NaN dans la colonne "Conviction"
    df_rules['Conviction'] = df_rules['Conviction'].replace([np.inf, -np.inf], np.nan)
    # Obtention de la conviction maximale de la dataframe (en excluant NaN)
    max_conviction = df_rules['Conviction'].max(skipna=True) + 1
    # Remplacement des NaN par la valeur maximale de conviction
    df_rules['Conviction'].fillna(max_conviction, inplace=True)  

    df_rules = df_rules.filter(["Antecedent","Conclusion", "Support", "Confiance", 'Lift', 'Conviction'], axis=1)   

       
    return df_rules

   

# In[4]:


def associationRuleVisualisation(df_rule,numberOfRule):
    ax2 = df.plot.scatter(x='Confiance',y='Support',colormap='viridis', title="Scatter plot of "+numberOfRule+" rules")



def classement_with_conv(df):
    """Cette fonction prends en entrée le chemin d'accès au fichier excel contenant les règles d'associations non redondantes, et procède au ranking de ces règles
    en utilisant la méthode TOPSIS. 
    Les crictères de rangement ici sont: le support, la confiance, le lift, la conviction et le taille des règles.
    Pour chaque crictère, on affecte un signal signal qui peut etre positif ou négatif(selon que l'on souhaite maximiser ou minimiser les valeurs sur ce crictère)
    Pour chaque crictère, on affecte également un poids (qui matérialise la force du crictère dans la décision finale). Cette affectation est faite automatiquement en utilisant une entropie.
    
    En sortie, on a une dataframe contenant les règles d'associations avec leurs score et rang
    """
    from scikitmcda.topsis import TOPSIS

    from scikitmcda.constants import MAX, MIN, LinearMinMax_, LinearSum_
    topsis = TOPSIS()
    import pandas as pd
    #df = pd.read_excel(input_file)
    df['Numero'] = [i+1 for i in range(len(df))]
    #df.rename(columns={'Unnamed: 0':'Numero'}, inplace=True)
    #df['Numero'] =  df['Numero'] +1
    
    df['NbItems'] = [len(df['Antecedent'].iloc[item].rstrip().split(" ")) for item in range(len(df))]
    df = df.filter(['Numero','Antecedent','Conclusion',"Support", "Confiance", 'Lift',  'Conviction', "NbItems" ], axis=1)  
    df = df.set_index(['Numero','Antecedent','Conclusion'])
    
    # Convert all columns from string to numeric (float)
    df = df.astype(float, errors='ignore')
    
    values = df.values.tolist()
    df = df.reset_index()
    
    topsis.dataframe(values, df['Numero'].tolist(), ["Support", "Confiance", 'Lift',  'Conviction', "NbItems"])
    
    #topsis.set_weights_manually([0.1, 0.1, 0.3,0.3,0.2])
    topsis.set_signals([MAX,MAX,MAX,MAX,MIN])
    topsis.set_weights_by_entropy(normalization_method_for_entropy=LinearSum_)
    
    topsis.decide(LinearMinMax_)
    
    Rule_decision = topsis.df_decision
    
    df['score']=topsis.df_decision['performance score'].tolist()
    df['rank']=topsis.df_decision['rank'].tolist()

    return df
    




