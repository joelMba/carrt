#!/usr/bin/env python
# coding: utf-8

"""
Module for preprocessing functions to be used before data preparation, modeling, training, etc.
"""

import ast

def data_preprocessing(df):
    """
    Preprocesses maintenance intervention data.
    Arguments:
        df : DataFrame - The data to preprocess.
    Returns:
        Preprocessed DataFrame.
    Main steps:
        - Filter closed interventions
        - Select relevant columns
        - Handle missing values
        - Convert columns to hours
        - Apply symbolic representation to numerical data
        - Remove unnecessary columns or columns with only one unique value
        - Remove duplicates and classes with only one occurrence
    """

    # Filter closed interventions
    df = df[df['status'] == 'closed']

    # Select relevant columns
    columns_to_keep = [
        "reference", "criticalLevel", "domain", "issuer.entity.label",
        "service.originalCode", "workType", "eventHistory",
        "responseTime", "processingDuration", "resolutionDuration",
        "technicalReason", "service.code"
    ]
    df = df.filter(columns_to_keep, axis=1)

    # Rename columns
    df.rename(columns={
        "issuer.entity.label": "provider",
        "service.originalCode": "providerCode",
        "domain": "client",
        "service.code": "clientCode"
    }, inplace=True)

    # Handle missing values
    df[["responseTime", "processingDuration", "resolutionDuration"]] = df[["responseTime", "processingDuration", "resolutionDuration"]].fillna(0)
    df["technicalReason"].fillna("monthly_visit", inplace=True)

    # Calculate total resolution duration and convert to hours
    df["resolutionDuration"] = df["responseTime"] + df["processingDuration"]
    time_columns = ["responseTime", "processingDuration", "resolutionDuration"]
    for col in time_columns:
        df[col] = df[col].apply(lambda x: x / (1000 * 60 * 60))
        df[col] = df[col].apply(symbolic_representation)

    # Correct workType values based on client codes
    corrective_cases = ["Panne chauffage", "Panne fuite d'eau", "Panne caisson VMC", "Pannes diverses", "Panne ECS"]
    preventive_cases = ["Visite entretien ventilation coll", "Visite entretien chauffage collectif", "Visite de contrôle/entretien"]
    df.loc[df['clientCode'].isin(corrective_cases), 'workType'] = 'corrective'
    df.loc[df['clientCode'].isin(preventive_cases), 'workType'] = 'preventive'

    # Add indicators based on event history
    event_indicators = [
        'acknowledged', 'start', 'done', 'planned', 'replanned', 'end', 'postponed', 'occupant_absent',
        'quote_request', 'requested', 'technical_issue', 'occupant_denial', 'client_planned', 'on_site',
        'missing_item', 'temporary_repair', 'canceled', 'partial_repair', 'formal_notice',
        'extension_request', 'solved', 'dismissed', 'updated', 'due', 'precisions_requested',
        'commented', 'non_contractual'
    ]
    for event in event_indicators:
        df[event] = df['eventHistory'].apply(lambda x: "True" if event in ast.literal_eval(x) else "False")

    # Rearrange columns and remove unnecessary ones
    client_code = df.pop('clientCode')
    df.drop(['eventHistory', 'providerCode'], axis=1, inplace=True)

    # Remove columns with only one unique value
    df = df[[col for col in df.columns if df[col].nunique() > 1]]

    # Remove duplicates
    df.set_index('reference', inplace=True)
    df = df.drop_duplicates().reset_index()

    # Remove classes with only one occurrence
    class_counts = df['clientCode'].value_counts()
    single_classes = class_counts[class_counts == 1].index
    df = df[~df['clientCode'].isin(single_classes)]

    # Rename and finalize
    df.rename(columns={"clientCode": "Class"}, inplace=True)

    return df


def symbolic_representation(time_duration):
    """
    Symbolically represents a duration in hours.
    Arguments:
        time_duration : float - Duration in hours.
    Returns:
        str - Symbolic category for the duration.
    """
    if time_duration <= 0:
        return "0"
    if time_duration <= 1:
        return "]0-1]"
    if time_duration <= 2:
        return "]1-2]"
    if time_duration <= 4:
        return "]2-4]"
    if time_duration <= 16:
        return "]4-16]"
    if time_duration <= 24:
        return "]16-24]"
    if time_duration <= 48:
        return "]24-48]"
    if time_duration <= 72:
        return "]48-72]"
    return "]72--["
