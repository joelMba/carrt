#!/usr/bin/env python
# coding: utf-8

"""
Module for feature extraction and dimension reduction using Brick ontology and maintenance activity data.
"""

import pandas as pd
import rdflib
from rdflib.namespace import RDFS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import re

def feature_extraction_from_brick(input_df):
    """
    Extract features from input descriptions using Brick ontology and maintenance activity ontology.
    Steps:
        - Parses Brick and maintenance ontologies to identify relevant classes and synonyms.
        - Processes text descriptions by cleaning, translating, and tokenizing.
        - Matches text with ontology classes and synonyms.
    Arguments:
        input_df : DataFrame - Input data with 'description' and 'tags' columns.
    Returns:
        DataFrame - Feature matrix with matched ontology classes as columns.
    """
    # Create an RDF graph and parse ontologies
    g = rdflib.Graph()
    g.parse("https://brickschema.org/schema/1.3/Brick.ttl", format="ttl")
    g.parse("maintenance_activity.owl", format="xml")

    # Query for Brick Equipment classes
    equipment_query = """
    SELECT DISTINCT ?class
    WHERE {
        ?class rdfs:subClassOf* brick:Equipment .
    }
    """
    equipment_classes = {str(row[0]).split("#")[-1] for row in g.query(equipment_query)}

    # Query for MaintenanceActivity classes and their synonyms
    maintenance_activity_query = """
    SELECT DISTINCT ?class ?synonym
    WHERE {
        ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
        ?class <http://www.industrialontologies.org/IOFAnnotationVocabulary/synonym> ?synonym .
    }
    """
    maintenance_activity_synonyms = {
        str(row[0]).split("#")[-1]: str(row[1]) for row in g.query(maintenance_activity_query)
    }

    # Classes to specifically check within Brick ontology
    brick_classes_to_check = {"Water", "Leak", "Room", "Solid", "Fluid", "Terminal_Unit"}

    # Extract references and filter relevant columns
    references = input_df['reference'].tolist()
    input_df = input_df.filter(['description', 'tags'], axis=1)

    # Define stopwords
    stopword_set = set(stopwords.words('english')).union(
        {'the', 'a', 'an', 'in', 'on', 'of', 'and', 'or', 'is', 'are'}
    )

    # Initialize output
    output_rows = []

    for _, row in input_df.iterrows():
        # Preprocess text
        text_data = f"{row['description']} {row['tags']}".replace('Code prestation', '')
        text_data = re.sub(r'\W+', ' ', text_data)
        text_data = text_data.replace('Robinetterie', 'eau').replace('Robinet', 'eau').replace('VMC', 'ventilateur')

        # Translate text to English
        translator = GoogleTranslator(source='auto', target='en')
        text_data = translator.translate(text_data)

        # Tokenize and clean text
        words = [w.lower() for w in word_tokenize(text_data) if w.lower() not in stopword_set]

        # Initialize class matches
        class_matches = {c: False for c in equipment_classes.union(maintenance_activity_synonyms.keys())}

        # Match words with ontology classes and synonyms
        for word in words:
            # Check Brick Equipment classes
            query_brick = f"""
            SELECT DISTINCT ?class
            WHERE {{
                ?class rdfs:label ?label .
                FILTER (lcase(str(?label)) = "{word}")
                ?class rdfs:subClassOf* brick:Equipment .
            }}
            """
            for row in g.query(query_brick):
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

            # Check Maintenance Activity classes
            query_maintenance = f"""
            SELECT DISTINCT ?class
            WHERE {{
                ?class rdfs:label ?label .
                FILTER (lcase(str(?label)) = "{word}")
                ?class rdfs:subClassOf* <http://www.semanticweb.org/maintenance-activity#MaintenanceActivity> .
            }}
            """
            for row in g.query(query_maintenance):
                class_name = str(row[0]).split("#")[-1]
                class_matches[class_name] = True

        # Match synonyms and split class names with underscores
        for class_name, synonym in maintenance_activity_synonyms.items():
            if synonym.lower() in words:
                class_matches[class_name] = True
        for class_name in equipment_classes.union(maintenance_activity_synonyms.keys()):
            if '_' in class_name and any(part.lower() in words for part in class_name.split('_')):
                class_matches[class_name] = True

        # Check specified Brick classes
        for brick_class in brick_classes_to_check:
            class_matches[brick_class] = brick_class.lower() in words

        # Append results
        output_rows.append(class_matches)

    # Convert results to a DataFrame
    output_df = pd.DataFrame(output_rows)
    output_df['reference'] = references

    # Keep only columns with at least one 'True' value
    true_columns = [col for col in output_df.columns if output_df[col].any()]
    return output_df[true_columns]


def dimension_reduction(df, minsup_tid):
    """
    Reduces the dimensionality of the DataFrame by removing low-frequency features.
    Arguments:
        df : DataFrame - Input DataFrame with extracted features.
        minsup_tid : float - Minimum support threshold (percentage of rows).
    Returns:
        DataFrame - Reduced DataFrame with filtered features.
    """
    # Drop columns with only one unique value
    df = df.loc[:, df.nunique() > 1]

    # Calculate minimum row count for the threshold
    nb_rows_min = len(df) * minsup_tid / 100

    # Filter out columns where 'True' count is below the threshold
    for column in df.columns:
        if df[column].astype(str).value_counts().get('True', 0) < nb_rows_min:
            df.drop(column, axis=1, inplace=True)
    
    return df
