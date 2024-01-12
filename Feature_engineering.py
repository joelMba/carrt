#!/usr/bin/env python
# coding: utf-8

# In[3]:


def feature_extraction_from_brick_Old(input_df):
    """
    Cette fonction prends en entrée une dataframe comportant une seule colonne comportant des données textuelles
    (Typiquement la description des opérations de maintenances des batiments).
    On utilise ensuite l'ontologie Brick pour extraire des informations pertinentes à partir de ces données textuelles.
    En sortie on a une dataframe qui comporte les caracteristiques extraites pour chaque élément de la dataframe d'entrée df.
    """
    
    import pandas as pd
    import rdflib
    from rdflib.namespace import RDFS
    from nltk.tokenize import word_tokenize
    from deep_translator import GoogleTranslator
    import re
    
    # Load the Brick ontology
    g = rdflib.Graph()
    g.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="ttl")
    
    classes_to_check = [
    "Air_Loop","Water_Loop",
   "Domestic_Hot_Water_System","Electrical_System", "Gas_System", "HVAC_System", "Heating_Ventilation_Air_Conditioning_System", "Lighting_System", "Safety_System", "Shading_System",
    "Camera", "Fire_Safety_Equipment", "Furniture", "Meter", "Motor","HVAC_Equipment", "Gas_Distribution", "Lighting_Equipment", "Motor", "PV_Panel", "Relay", "Safety_Equipment", "Security_Equipment", "Shading_Equipment", "Solar_Thermal_Collector", "Valve", 
    "Water_Distribution","Water_Heater","Weather_Station",
    "AHU", "Air_Handler_Unit", "Air_Handling_Unit", "Air_Plenum", "Boiler", "Bypass_Valve", "CRAC", "CRAH", "Chiller", "Fan", "Filter",
    "Fume_Hood", "HVAC_Valve", "HX", "Heat_Exchanger", "Heating_Valve", "Hot_Deck", "Humidifier", "Isolation_Valve", "Pump", "Space_Heater",
    "Steam_Valve", "Terminal_Unit", "Thermostat",
    "Common_Space", "Entrance", "Gatehouse", "Media_Hot_Desk", "Parking_Space", "Room", "Ticketing_Booth", "Tunnel", "Vertical_Space","Water_Tank",
     "Leak"
    ]
    
    # Inputs data
    input_df = input_df[input_df['status']=='closed']
    references = input_df['reference'].tolist()
    input_df = input_df.filter(['description'], axis=1)
    
    # Check each text in the input DataFrame for matches
    output_rows = []
    for _, row in input_df.iterrows():
        # Pre-process the text data
        text_data = str(row['description'])
        text_data = text_data.replace('Code prestation', '')
        text_data = re.sub(r'\W+', ' ', text_data)
        # Translate text from French to English
        translator = GoogleTranslator(source='auto', target='en')
        text_data = translator.translate(text_data)
        words = word_tokenize(text_data)

        class_matches = {c: False for c in classes_to_check}
        
        for word in words:
            query = f"""
                SELECT DISTINCT ?class ?parent ?equivalent
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER regex(lcase(str(?label)), "{word.lower()}", "i")

                    ?class rdfs:subClassOf* ?parent .
                    ?class owl:equivalentClass* ?equivalent .
                }}
            """
            results = g.query(query)

            for row in results:
                class_name = str(row['class']).split('#')[-1]
                parent_name = str(row['parent']).split('#')[-1]
                equivalent_name = str(row['equivalent']).split('#')[-1]
                if class_name in classes_to_check:
                    class_matches[class_name] = True
                if parent_name in classes_to_check:
                    class_matches[parent_name] = True
                if equivalent_name in classes_to_check:
                    class_matches[equivalent_name] = True

        output_rows.append(class_matches)

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(output_rows)
    output_df['reference'] = references

    return output_df


# In[6]:


def feature_extraction_from_brick_old(input_df):
    """
    Cette fonction prends en entrée une dataframe comportant une seule colonne comportant des données textuelles
    (Typiquement la description des opérations de maintenances des batiments).
    On utilise ensuite l'ontologie Brick pour extraire des informations pertinentes à partir de ces données textuelles.
    En sortie on a une dataframe qui comporte les caracteristiques extraites pour chaque élément de la dataframe d'entrée df.
    """
    
    import pandas as pd
    import rdflib
    from rdflib.namespace import RDFS
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from deep_translator import GoogleTranslator
    import re
    
    # Load the Brick ontology
    g = rdflib.Graph()
    g.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="ttl")
    
    classes_to_check = [
    "Air_Loop","Water_Loop",
   "Domestic_Hot_Water_System","Electrical_System", "Gas_System", "HVAC_System", "Heating_Ventilation_Air_Conditioning_System", "Lighting_System", "Safety_System", "Shading_System",
    "Camera", "Fire_Safety_Equipment", "Furniture", "Meter", "Motor","HVAC_Equipment", "Gas_Distribution", "Lighting_Equipment", "Motor", "PV_Panel", "Relay", "Safety_Equipment", "Security_Equipment", "Shading_Equipment", "Solar_Thermal_Collector", "Valve", 
    "Water_Distribution","Water_Heater","Weather_Station",
    "AHU", "Air_Handler_Unit", "Air_Handling_Unit", "Air_Plenum", "Boiler", "Bypass_Valve", "CRAC", "CRAH", "Chiller", "Fan", "Filter",
    "Fume_Hood", "HVAC_Valve", "HX", "Heat_Exchanger", "Heating_Valve", "Hot_Deck", "Humidifier", "Isolation_Valve", "Pump", "Space_Heater",
    "Steam_Valve", "Terminal_Unit", "Thermostat",
    "Fluid", "Solid",
    "Alarm", "Command", "Parameter", "Sensor", "Setpoint", "Status",
    "Common_Space", "Entrance", "Gatehouse", "Media_Hot_Desk", "Parking_Space", "Room", "Ticketing_Booth", "Tunnel", "Vertical_Space","Water_Tank",
     "Leak"
    ]
    
    classes_to_check = [
    "Air_Loop","Water_Loop",
   "Domestic_Hot_Water_System","Electrical_System", "Gas_System", "Heating_Ventilation_Air_Conditioning_System", "Lighting_System", "Safety_System", "Shading_System",
    "Camera", "Fire_Safety_Equipment", "Furniture", "Meter", "Motor","HVAC_Equipment", "Gas_Distribution", "Lighting_Equipment", "Motor", "PV_Panel", "Relay", "Safety_Equipment", "Security_Equipment", "Shading_Equipment", "Solar_Thermal_Collector", "Valve", 
    "Water_Distribution","Water_Heater","Weather_Station",
    "AHU", "Air_Handler_Unit", "Air_Handling_Unit", "Air_Plenum", "Boiler", "Bypass_Valve", "CRAC", "CRAH", "Chiller", "Cold_Deck", 
    "Compressor", "Computer_Room_Air_Conditioning", "Computer_Room_Air_Handler", "Condenser", "Cooling_Tower", "Cooling_Valve", "Damper", "Fan", "Filter",
    "Fume_Hood", "HVAC_Valve", "HX", "Heat_Exchanger", "Heating_Valve", "Hot_Deck", "Humidifier", "Isolation_Valve", "Pump", "Space_Heater",
    "Steam_Valve", 
    "Terminal_Unit", "Air_Diffuser", "CAV", "Chilled_Beam", "Constant_Air_Volume_Box", "VAV", "Radiator", "Radiant_Panel", "Thermostat",
    "Fluid", "Gas", "Air","CO", "CO2", "Natural_Gas", "Steam", "Liquid", "Gasoline", "Glycol", "Liquid_CO2", "Oil", "Water", "Chilled_Water", "Condenser_Water", 
    "Domestic_Water", "Hot_Water", "Refrigerant",
    "Solid", "Frost", "Hail", "Ice", "Soil",
    "Alarm", "Command", "Parameter", "Sensor", "Setpoint", "Status",
    "Common_Space", "Entrance", "Gatehouse", "Media_Hot_Desk", "Parking_Space", "Room", "Ticketing_Booth", "Tunnel", "Vertical_Space","Water_Tank",
     "Leak"
    ]
    
    # Inputs data
    input_df = input_df[input_df['status']=='closed']
    references = input_df['reference'].tolist()
    input_df = input_df.filter(['description','tags'], axis=1)
    
    # Define the set of words to remove from the input text    
    stopwords = set(stopwords.words('english')) | set(['the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'is', 'are'])
    
    # Check each text in the input DataFrame for matches
    output_rows = []
    
    for _, row in input_df.iterrows():
        # Pre-process the text data
        text_data = str(row['description']) + " " + str(row['tags'])
        text_data = text_data.replace('Code prestation', '')
        text_data = text_data.replace('Robinetterie', 'eau')
        text_data = text_data.replace('Robinet', 'eau')
        text_data = text_data.replace('robinet', 'eau')
        text_data = text_data.replace('VMC', 'ventilateur')
        text_data = text_data.replace('vmc', 'ventilateur')
        text_data = re.sub(r'\W+', ' ', text_data)
        # Translate text from French to English
        translator = GoogleTranslator(source='auto', target='en')
        text_data = translator.translate(text_data)
        
        # Tokenize the input text and remove stopwords
        words = [w.lower() for w in word_tokenize(text_data) if w.lower() not in stopwords]

        
        class_matches = {c: False for c in classes_to_check}
        
        for word in words:           
            
            query = f"""
                SELECT DISTINCT ?class ?parent ?equivalent
                WHERE {{
                    {{

                    }} UNION {{
                        ?class rdfs:label ?label .
                        FILTER (lcase(str(?label)) = "{word.lower()}")
                        ?class rdfs:subClassOf* ?parent .
                        ?class owl:equivalentClass* ?equivalent .
                    }}                                      
                }}
            """
            results = g.query(query)

            for row in results:
                class_name = str(row['class']).split('#')[-1]
                parent_name = str(row['parent']).split('#')[-1]
                equivalent_name = str(row['equivalent']).split('#')[-1]
                if class_name in classes_to_check:
                    class_matches[class_name] = True
                if parent_name in classes_to_check:
                    class_matches[parent_name] = True
                if equivalent_name in classes_to_check:
                    class_matches[equivalent_name] = True

        output_rows.append(class_matches)

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(output_rows)
    output_df['reference'] = references

    return output_df



def feature_extraction_from_brick_old_2(input_df):
    
    import pandas as pd
    import rdflib
    from rdflib.namespace import RDFS
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from deep_translator import GoogleTranslator
    import re
    # Create an RDF graph
    g = rdflib.Graph()

    # Parse the Brick ontology
    g.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="ttl")

    # Parse the Maintenance Activity ontology from the local file
    g.parse("maintenance_activity.owl", format="xml")

    # Define the Brick "Equipment" class and its subclasses
    equipment_query = """
    SELECT DISTINCT ?class
    WHERE {
        ?class rdfs:subClassOf* brick:Equipment .
    }
    """

    equipment_classes = set()
    for row in g.query(equipment_query):
        class_name = str(row[0]).split("#")[-1]
        equipment_classes.add(class_name)

    # Define the MaintenanceActivity classes and their synonyms
    maintenance_activity_query = """
    SELECT DISTINCT ?class ?synonym
    WHERE {
        ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
        ?class <http://www.industrialontologies.org/IOFAnnotationVocabulary/synonym> ?synonym .
    }
    """

    maintenance_activity_synonyms = {}
    for row in g.query(maintenance_activity_query):
        class_name = str(row[0]).split("#")[-1]
        synonym = str(row[1])
        maintenance_activity_synonyms[class_name] = synonym

    # Inputs data
    #input_df = input_df[input_df['status'] == 'closed']
    references = input_df['reference'].tolist()
    input_df = input_df.filter(['description', 'tags'], axis=1)

    # Define the set of words to remove from the input text
    stopword = set(stopwords.words('english')) | set(['the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'is', 'are'])

    # Check each text in the input DataFrame for matches
    output_rows = []

    for _, row in input_df.iterrows():
        # Pre-process the text data
        text_data = str(row['description']) + " " + str(row['tags'])
        text_data = text_data.replace('Code prestation', '')
        text_data = text_data.replace('Robinetterie', 'eau')
        text_data = text_data.replace('Robinet', 'eau')
        text_data = text_data.replace('robinet', 'eau')
        text_data = text_data.replace('VMC', 'ventilateur')
        text_data = text_data.replace('vmc', 'ventilateur')
        text_data = re.sub(r'\W+', ' ', text_data)
        # Translate text from French to English
        translator = GoogleTranslator(source='auto', target='en')
        text_data = translator.translate(text_data)

        # Tokenize the input text and remove stopwords
        words = [w.lower() for w in word_tokenize(text_data) if w.lower() not in stopword]

        class_matches = {c: False for c in equipment_classes.union(maintenance_activity_synonyms.keys())}

        for word in words:
            # Check Brick Equipment classes
            query_brick = f"""
                SELECT DISTINCT ?class
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER (lcase(str(?label)) = "{word.lower()}")
                    ?class rdfs:subClassOf* brick:Equipment .
                }}
            """
            results_brick = g.query(query_brick)

            # Check Maintenance Activity classes
            query_maintenance_activity = f"""
                SELECT DISTINCT ?class
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER (lcase(str(?label)) = "{word.lower()}")
                    ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
                }}
            """
            results_maintenance_activity = g.query(query_maintenance_activity)

            for row in results_brick:
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

            for row in results_maintenance_activity:
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

        # Check synonyms for Maintenance Activity classes
        for class_name, synonym in maintenance_activity_synonyms.items():
            if synonym.lower() in words:
                class_matches[class_name] = True

        # Check if at least one word in class name with underscores matches any word in text_data
        for class_name in equipment_classes.union(maintenance_activity_synonyms.keys()):
            if '_' in class_name:
                class_words = class_name.split('_')
                for word in words:
                    if any(class_word == word.lower() for class_word in class_words):
                        class_matches[class_name] = True
                        break

        output_rows.append(class_matches)

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(output_rows)
    output_df['reference'] = references

    # Filter columns with 'True' values
    true_columns = [col for col in output_df.columns if output_df[col].any()]
    return output_df[true_columns]
# In[6]:


import pandas as pd
import rdflib
from rdflib.namespace import RDFS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import re

def feature_extraction_from_brick_old3(input_df):
    # Create an RDF graph
    g = rdflib.Graph()

    # Parse the Brick ontology
    g.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="ttl")

    # Parse the Maintenance Activity ontology from the local file
    g.parse("maintenance_activity.owl", format="xml")

    # Define the Brick "Equipment" class and its subclasses
    equipment_query = """
    SELECT DISTINCT ?class
    WHERE {
        ?class rdfs:subClassOf* brick:Equipment .
    }
    """

    equipment_classes = set()
    for row in g.query(equipment_query):
        class_name = str(row[0]).split("#")[-1]
        equipment_classes.add(class_name)

    # Manually add the additional class terms
    additional_classes = ["Water", "Leak", "Room", "Solid", "Fluid", "Terminal_Unit"]
    equipment_classes.update(additional_classes)

    # Define the MaintenanceActivity classes and their synonyms
    maintenance_activity_query = """
    SELECT DISTINCT ?class ?synonym
    WHERE {
        ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
        ?class <http://www.industrialontologies.org/IOFAnnotationVocabulary/synonym> ?synonym .
    }
    """

    maintenance_activity_synonyms = {}
    for row in g.query(maintenance_activity_query):
        class_name = str(row[0]).split("#")[-1]
        synonym = str(row[1])
        maintenance_activity_synonyms[class_name] = synonym

    # Inputs data
    #input_df = input_df[input_df['status'] == 'closed']
    references = input_df['reference'].tolist()
    input_df = input_df.filter(['description', 'tags'], axis=1)

    # Define the set of words to remove from the input text
    stopword = set(stopwords.words('english')) | set(['the', 'a', 'an', 'in', 'on', 'of', 'and', ' or', 'is', 'are'])

    # Check each text in the input DataFrame for matches
    output_rows = []

    for _, row in input_df.iterrows():
        # Pre-process the text data
        text_data = str(row['description']) + " " + str(row['tags'])
        text_data = text_data.replace('Code prestation', '')
        text_data = text_data.replace('Robinetterie', 'eau')
        text_data = text_data.replace('Robinet', 'eau')
        text_data = text_data.replace('robinet', 'eau')
        text_data = text_data.replace('VMC', 'ventilateur')
        text_data = text_data.replace('vmc', 'ventilateur')
        text_data = re.sub(r'\W+', ' ', text_data)
        # Translate text from French to English
        translator = GoogleTranslator(source='auto', target='en')
        text_data = translator.translate(text_data)

        # Tokenize the input text and remove stopwords
        words = [w.lower() for w in word_tokenize(text_data) if w.lower() not in stopword]

        class_matches = {c: False for c in equipment_classes.union(maintenance_activity_synonyms.keys())}

        for word in words:
            # Check Brick Equipment classes
            query_brick = f"""
                SELECT DISTINCT ?class
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER (lcase(str(?label)) = "{word.lower()}")
                    ?class rdfs:subClassOf* brick:Equipment .
                }}
            """
            results_brick = g.query(query_brick)

            # Check Maintenance Activity classes
            query_maintenance_activity = f"""
                SELECT DISTINCT ?class
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER (lcase(str(?label)) = "{word.lower()}")
                    ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
                }}
            """
            results_maintenance_activity = g.query(query_maintenance_activity)

            for row in results_brick:
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

            for row in results_maintenance_activity:
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

        # Check synonyms for Maintenance Activity classes
        for class_name, synonym in maintenance_activity_synonyms.items():
            if synonym.lower() in words:
                class_matches[class_name] = True

        # Check if at least one word in class name with underscores matches any word in text_data
        for class_name in equipment_classes.union(maintenance_activity_synonyms.keys()):
            if '_' in class_name:
                class_words = class_name.split('_')
                for word in words:
                    if any(class_word == word.lower() for class_word in class_words):
                        class_matches[class_name] = True
                        break

        output_rows.append(class_matches)

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(output_rows)
    output_df['reference'] = references

    # Filter columns with 'True' values
    true_columns = [col for col in output_df.columns if output_df[col].any()]
    return output_df[true_columns]

import pandas as pd
import rdflib
from rdflib.namespace import RDFS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import re

def feature_extraction_from_brick(input_df):
    # Create an RDF graph
    g = rdflib.Graph()

    # Parse the Brick ontology
    g.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="ttl")

    # Parse the Maintenance Activity ontology from the local file
    g.parse("maintenance_activity.owl", format="xml")

    # Define the Brick "Equipment" class and its subclasses
    equipment_query = """
    SELECT DISTINCT ?class
    WHERE {
        ?class rdfs:subClassOf* brick:Equipment .
    }
    """

    equipment_classes = set()
    for row in g.query(equipment_query):
        class_name = str(row[0]).split("#")[-1]
        equipment_classes.add(class_name)

    # Define the MaintenanceActivity classes and their synonyms
    maintenance_activity_query = """
    SELECT DISTINCT ?class ?synonym
    WHERE {
        ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
        ?class <http://www.industrialontologies.org/IOFAnnotationVocabulary/synonym> ?synonym .
    }
    """

    maintenance_activity_synonyms = {}
    for row in g.query(maintenance_activity_query):
        class_name = str(row[0]).split("#")[-1]
        synonym = str(row[1])
        maintenance_activity_synonyms[class_name] = synonym

    # Classes to check for in Brick
    brick_classes_to_check = ["Water", "Leak", "Room", "Solid", "Fluid", "Terminal_Unit"]

    # Inputs data
    references = input_df['reference'].tolist()
    input_df = input_df.filter(['description', 'tags'], axis=1)

    # Define the set of words to remove from the input text
    stopword = set(stopwords.words('english')) | set(['the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'is', 'are'])

    # Check each text in the input DataFrame for matches
    output_rows = []

    for _, row in input_df.iterrows():
        # Pre-process the text data
        text_data = str(row['description']) + " " + str(row['tags'])
        text_data = text_data.replace('Code prestation', '')
        text_data = text_data.replace('Robinetterie', 'eau')
        text_data = text_data.replace('Robinet', 'eau')
        text_data = text_data.replace('robinet', 'eau')
        text_data = text_data.replace('VMC', 'ventilateur')
        text_data = text_data.replace('vmc', 'ventilateur')
        text_data = re.sub(r'\W+', ' ', text_data)
        # Translate text from French to English
        translator = GoogleTranslator(source='auto', target='en')
        text_data = translator.translate(text_data)

        # Tokenize the input text and remove stopwords
        words = [w.lower() for w in word_tokenize(text_data) if w.lower() not in stopword]

        class_matches = {c: False for c in equipment_classes.union(maintenance_activity_synonyms.keys())}

        for word in words:
            # Check Brick Equipment classes
            query_brick = f"""
                SELECT DISTINCT ?class
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER (lcase(str(?label)) = "{word.lower()}")
                    ?class rdfs:subClassOf* brick:Equipment .
                }}
            """
            results_brick = g.query(query_brick)

            # Check Maintenance Activity classes
            query_maintenance_activity = f"""
                SELECT DISTINCT ?class
                WHERE {{
                    ?class rdfs:label ?label .
                    FILTER (lcase(str(?label)) = "{word.lower()}")
                    ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
                }}
            """
            results_maintenance_activity = g.query(query_maintenance_activity)

            for row in results_brick:
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

            for row in results_maintenance_activity:
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

        # Check synonyms for Maintenance Activity classes
        for class_name, synonym in maintenance_activity_synonyms.items():
            if synonym.lower() in words:
                class_matches[class_name] = True

        # Check if at least one word in class name with underscores matches any word in text_data
        for class_name in equipment_classes.union(maintenance_activity_synonyms.keys()):
            if '_' in class_name:
                class_words = class_name.split('_')
                for word in words:
                    if any(class_word == word.lower() for class_word in class_words):
                        class_matches[class_name] = True
                        break

        # Check if any of the specified Brick classes exist in the text
        for brick_class in brick_classes_to_check:
            if brick_class.lower() in words:
                class_matches[brick_class] = True
            else : 
                class_matches[brick_class] = False

        output_rows.append(class_matches)

    # Convert the results to a DataFrame
    output_df = pd.DataFrame(output_rows)
    output_df['reference'] = references

    # Filter columns with 'True' values
    true_columns = [col for col in output_df.columns if output_df[col].any()]
    return output_df[true_columns]



# In[ ]:

def dimension_reduction(df,minsup_tid):
    """ Cette fonction permet de réduire la dimension de notre dataframe.
    la variable df représente les features extraits de l'ontologies, et donc les valeurs sont des booleens 'True' et 'False' 
    La fonction prends également en entrée le minsup tid.
    Par la suite nous supprimons les colonnes avec un tid inférieur au min"""
    #Suppression des colonnes avec une seule modalité
    for item in df.columns.tolist() :
        if (len(df[item].unique().tolist())==1) :
            df = df.drop([item], axis=1)
                
    nb_rows_min = (len(df)*minsup_tid)/100
    df = df.astype(str)
    columns = df.columns.tolist()
    for item in columns:
        counts = df[item].value_counts()
        nb_rows = counts['True']
        if(nb_rows < nb_rows_min):
            df = df.drop([item], axis=1)
    return df




