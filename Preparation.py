#!/usr/bin/env python
# coding: utf-8

"""
Ce module contient des fonctions pour la préparation des données, afin de les adapter au format d'entrée de l'algorithme d'apprentissage
"""

def dataConversionToSpmfTxt(df,chemin):
    """ Cette fonction prend en entrée les opération/interventions prétraitées et les transforme en au format spmf.
    - La fonction prends également en entrée le chemin d'acces au fichier sauvergarde les données transformées """
    for item in df.columns.tolist() :
        df[item] = [item + "=" + str(elt) for elt in df[item].values]
    chemin = chemin
    liste = listOfDistinctElt(df)
    dico = recodification(liste)
    df1 = dataCodification (df, dico)
    exportDataToSPMFTxt(df1, dico, chemin)


# In[4]:


def listOfDistinctElt (df):
    listColumns = df.columns.tolist()
    resultList = []
    for item in listColumns:
        resultList += df[item].unique().tolist()
    return resultList


# In[5]:


def recodification(liste):
    """Fonction de codification des évènements
    La fonction prends en entrée une liste (évenements de la base) et retourne un dictionnaire comportant 
    les évènements codifiés"""
    return {k: v for v, k in enumerate(liste,1)}


# In[6]:


def dataCodification (df, dico):
    for item in df.columns.tolist():
        df = df.replace({item:dico})
    return df


# In[7]:


def exportDataToSPMFTxt(df, dico, chemin):
        records = []
        for i in range(0,len(df)):
            records.append([str(df.values[i,j]) for j in range(0, len(df.columns.tolist()))])

        #with open(r'Donnees/test.txt', 'w') as fp:
        with open(chemin, 'w') as fp:
            fp.write("%s\n" % "@CONVERTED_FROM_TEXT") 
            for elt in dico:
                param = ""
                param = "@ITEM=" + str(dico[elt]) + "=" + elt
                fp.write("%s\n" % param)
            for elt in records:
                sequence = ""
                for item in elt:
                    sequence = sequence + item + " "
                sequence = "".join(sequence.rstrip()) #delete a space at the end of string
                    # write each item on a new line
                fp.write("%s\n" % sequence)


# In[ ]:


def delete_False_Item (file_path) :
    items_to_remove = []

    with open(file_path, "r") as input_file:
        contents = input_file.read()
        lines = contents.splitlines()

        for i in range(len(lines)):
            # Check if the line starts with "@"
            if lines[i].startswith("@"):
                items = lines[i].split("=")
                if items[-1]=="False":
                    items_to_remove.append(items[1])


    with open(file_path, "r") as input_file:
        contents = input_file.read()
        lines = contents.splitlines()

        for i in range(len(lines)):
            # Check if the line starts with "@"
            if not lines[i].startswith("@"):
                items = lines[i].split(" ")
                items = [item for item in items if item not in items_to_remove]
                lines[i] = " ".join(items)

    with open(file_path, "w") as output_file:
        output_file.write("\n".join(lines))

