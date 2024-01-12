#!/usr/bin/env python
# coding: utf-8

# In[10]:


"""Ce module comporte les fonctions de préprocessing des données, à faire en amont des étapes de préparation, modelisation, entrainement,...etc. """


def data_preprocessing_old(df):

       """Cette fonction prends en entrée les opérations de maintenance d'un prestataire pour un client donnée, et retourne la dataframe avec les données transformées conformement à nos besoins
          Ci-dessous la liste non exhaustive des taches effectuées:  
            - Sélection des interventions cloturées (En rappelant au passage que nous ne traitons que les interventions cloturées)
            - Sélection des colonnes pertinentes 
            - Remplacer les données manquantes par 0 (pour les données numériques)
            - Conversion en heures des colonnes "responseTime", "processingDuration", "resolutionDuration"
            - Représentation symbolique des données numérique
            - Suppression des colonnes avec une seule modalité
            - Suppression des doublons
            - ... Etc."""
       
       #On traite uniquement les interventions cloturees
       df = df[df['status']=='closed']
       # Sélection des colonnes pertinentes
       df = df.filter(["reference","criticalLevel","domain","issuer.entity.label","service.originalCode", "workType", "eventHistory", "responseTime", "processingDuration", "resolutionDuration", "technicalReason", "service.code" ], axis=1)   
       df["criticalLevel"] = df["criticalLevel"].astype(str)
       # Renommage des colonnes
       df.rename(columns={"issuer.entity.label":"prestataire","service.originalCode":"codePrestataire","domain":"client","service.code":"codeClient"}, inplace=True)
       # Données manquantes
       # Colonnes numérique
       # Pourquoi remplacer les valeurs manquantes des colonnes numériques par zéro ?
       df[["responseTime","processingDuration","resolutionDuration"]] = df[["responseTime","processingDuration","resolutionDuration"]].fillna(0)
       # Colonnes non numériques
       # Remplacer par une valeur par défaut
       df[["technicalReason"]] = df[["technicalReason"]].fillna("visite_mensuelle")
       df["resolutionDuration"] = df["responseTime"] + df["processingDuration"]
       # conversion en heures des colonnes "responseTime", "processingDuration", "resolutionDuration"
       df["responseTime"] = [item/(1000*60*60) for item in df.responseTime]
       df["processingDuration"] = [item/(1000*60*60) for item in df.processingDuration]
       df["resolutionDuration"] = [item/(1000*60*60) for item in df.resolutionDuration]
       # Représentation symbolique des données numérique
       df["responseTime"] = [symbolic_representation(item) for item in df.responseTime]
       df["processingDuration"] = [symbolic_representation(item) for item in df.processingDuration]
       df["resolutionDuration"] = [symbolic_representation(item) for item in df.resolutionDuration]
       
       #codes taggés avec de mauvais workType
       df.loc[df['codeClient'] == "Panne chauffage", 'workType'] = 'corrective'
       df.loc[df['codeClient'] == "Panne fuite d'eau", 'workType'] = 'corrective'
       df.loc[df['codeClient'] == "Panne caisson VMC", 'workType'] = 'corrective'       
       df.loc[df['codeClient'] == "Pannes diverses", 'workType'] = 'corrective'
       df.loc[df['codeClient'] == "Panne ECS", 'workType'] = 'corrective'
       df.loc[df['codeClient'] == "Visite entretien ventilation coll", 'workType'] = 'preventive'
       df.loc[df['codeClient'] == "Visite entretien chauffage collectif", 'workType'] = 'preventive'
       df.loc[df['codeClient'] == "Visite de contrôle/entretien", 'workType'] = 'preventive'
       
       #On remplace par les worktypes initiaux paramétrés par les clients
       #df['workType'] = df['service.workType'].tolist()
       
       #df = df.drop(['workType'], axis=1)

       #suppression colonne codeClient et rajout en fin de dataframe
       codeClient = df['codeClient'].tolist()
       df = df.drop(['codeClient'], axis=1)
       import ast
       for item in ['acknowledged', 'start', 'done', 'planned', 'replanned', 'end','postponed', 'occupant_absent', 'quote_request', 'requested',
          'technical_issue', 'occupant_denial', 'client_planned', 'on_site','missing_item', 'temporary_repair', 'canceled', 'partial_repair',
          'formal_notice', 'extension_request', 'solved', 'dismissed','updated', 'due', 'precisions_requested', 'commented','non_contractual'] :
           df[item] = ["True" if item in ast.literal_eval(a) else "False" for a in df['eventHistory']]    

       df['codeClient'] = codeClient

       #Suppression des colonnes avec une seule modalité
       for item in df.columns.tolist() :
           if (len(df[item].unique().tolist())==1) :
               df = df.drop([item], axis=1)

       #Suppression des colonnes inutiles        
       df = df.drop(['eventHistory'], axis=1)
       #df = df.drop(['prestataire'], axis=1)
       df = df.drop(['codePrestataire'], axis=1)
       #Suppression des doublons
       df.set_index('reference', inplace=True)
       df.drop_duplicates(inplace=True)
       df = df.reset_index()
       
       # Compter le nombre d'occurrences de chaque classe
       class_counts = df['codeClient'].value_counts()

       # Identifier les classes avec un seul élément
       single_element_classes = class_counts[class_counts == 1].index.tolist()

       # Supprimer les lignes correspondant à ces classes
       df = df[~df['codeClient'].isin(single_element_classes)]
       
       df.rename(columns={"codeClient":"Class"}, inplace=True)
       
       
       
       # Drop rows with null values
       df.set_index('reference', inplace=True)
       df = df.dropna()
       df = df.reset_index()
       
       #df = df.drop(['reference'], axis=1)

       return df


# In[17]:


def symbolic_representation (time_duration) :
        """    Fonction pour la Représentation symbolique (catégoristaion) des données numérique, responseTime, processingDuration, resolutionDuration
               La fonction prends en entrée du durée quelconque (estimée en heure) et retourne sa représentatin symbolique, qui peut etre:
                    - "0" si la durée est de 0
                    - "]0-1]" si durée inférieure à 1 heure
                    - "]1-2]" si durée comprise entre 1 et 2 heures
                    - "]2-4]"
                    - "]4-16]"
                    - "]16-24]"
                    - "]24-48]"
                    - "]48-72]"
                    - "]72--[" pour les durée au delàs de 72 heures
        """
        result = 0
        if (time_duration == 0.0 or time_duration < 0) :
            result = "0"
        if (time_duration > 0.0 and time_duration <= 1) :
            result = "]0-1]"
        if (time_duration > 1 and time_duration <= 2) :
            result = "]1-2]"
        if (time_duration > 2 and time_duration <= 4) :
            result = "]2-4]"
        if (time_duration > 4 and time_duration <= 16) :
            result = "]4-16]"
        if (time_duration > 16 and time_duration <= 24) :
            result = "]16-24]"
        if (time_duration > 24 and time_duration <= 48) :
            result = "]24-48]"
        if (time_duration > 48 and time_duration <= 72) :
            result = "]48-72]"
        if (time_duration > 72) :
            result = "]72--["

        return result


# In[ ]:




