#!/usr/bin/env python
# coding: utf-8

"""
Module for preprocessing functions to prepare data for analysis, including transformation, filtering, and categorization.
"""

import ast

def data_preprocessing_old(df):
    """
    Preprocesses input data by filtering and transforming it for analysis.
    Steps:
        - Filters closed interventions.
        - Selects and renames relevant columns.
        - Handles missing values in numerical and non-numerical columns.
        - Converts durations to hours and categorizes them symbolically.
        - Processes event history for frequent events.
        - Removes unnecessary columns and duplicates.
    Arguments:
        df : DataFrame - Input data containing intervention records.
    Returns:
        DataFrame - Transformed and cleaned data.
    """
    # Filter for closed interventions only
    df = df[df['status'] == 'closed']

    # Select relevant columns
    df = df.filter([
        "reference", "criticalLevel", "domain", "issuer.entity.label", 
        "service.originalCode", "eventHistory", "responseTime", 
        "processingDuration", "resolutionDuration", "technicalReason", 
        "service.code"
    ], axis=1)

    # Rename columns for clarity
    df.rename(columns={
        "issuer.entity.label": "serviceProvider",
        "service.originalCode": "providerCode",
        "domain": "client",
        "service.code": "clientCode"
    }, inplace=True)

    # Handle missing values
    # Fill missing numerical values with 0
    df[["responseTime", "processingDuration", "resolutionDuration"]] = \
        df[["responseTime", "processingDuration", "resolutionDuration"]].fillna(0)

    # Fill missing non-numerical values with a default value
    df["technicalReason"] = df["technicalReason"].fillna("monthly_visit")

    # Update resolutionDuration to be the sum of responseTime and processingDuration
    df["resolutionDuration"] = df["responseTime"] + df["processingDuration"]

    # Convert durations from milliseconds to hours
    df["responseTime"] = df["responseTime"] / (1000 * 60 * 60)
    df["processingDuration"] = df["processingDuration"] / (1000 * 60 * 60)
    df["resolutionDuration"] = df["resolutionDuration"] / (1000 * 60 * 60)

    # Categorize numerical durations symbolically
    df["responseTime"] = df["responseTime"].apply(symbolic_representation)
    df["processingDuration"] = df["processingDuration"].apply(symbolic_representation)
    df["resolutionDuration"] = df["resolutionDuration"].apply(symbolic_representation)

    # Temporarily store the clientCode column for reordering later
    client_code_column = df.pop("clientCode")

    # Process eventHistory for frequent events
    frequent_events = [
        'acknowledged', 'start', 'done', 'planned', 'replanned', 'end',
        'postponed', 'occupant_absent', 'quote_request', 'requested',
        'technical_issue', 'occupant_denial', 'client_planned', 'on_site',
        'missing_item', 'temporary_repair', 'canceled', 'partial_repair',
        'formal_notice', 'extension_request', 'solved', 'dismissed',
        'updated', 'due', 'precisions_requested', 'commented', 'non_contractual'
    ]
    for event in frequent_events:
        df[event] = df["eventHistory"].apply(
            lambda x: "True" if event in ast.literal_eval(x) else "False"
        )

    # Add clientCode back to the DataFrame
    df["clientCode"] = client_code_column

    # Remove columns with only one unique value
    df = df.loc[:, df.nunique() > 1]

    # Drop unnecessary columns
    df.drop(columns=["providerCode", "eventHistory"], inplace=True)

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
