#!/usr/bin/env python
# coding: utf-8

"""
This module contains functions for preparing data and converting it into the SPMF input format for machine learning algorithms.
"""

def data_conversion_to_spmf_txt(df, output_path):
    """
    Converts preprocessed intervention/operation data into SPMF format.
    Arguments:
        df : DataFrame - The preprocessed data.
        output_path : str - The file path to save the transformed data.
    """
    # Format each value as "column=value"
    for column in df.columns:
        df[column] = df[column].apply(lambda x: f"{column}={x}")

    # Generate a list of distinct elements and their encoded mapping
    distinct_elements = get_distinct_elements(df)
    encoding_dict = encode_elements(distinct_elements)

    # Encode data using the generated mapping
    encoded_df = encode_data(df, encoding_dict)

    # Export the encoded data to an SPMF-compatible text file
    export_data_to_spmf_txt(encoded_df, encoding_dict, output_path)


def get_distinct_elements(df):
    """
    Creates a list of unique elements from the DataFrame.
    Arguments:
        df : DataFrame - The data.
    Returns:
        list - A list of distinct elements across all columns.
    """
    unique_elements = []
    for column in df.columns:
        unique_elements.extend(df[column].unique())
    return unique_elements


def encode_elements(elements):
    """
    Encodes events into numeric format.
    Arguments:
        elements : list - A list of events to encode.
    Returns:
        dict - A dictionary mapping each event to a unique numeric code.
    """
    return {element: idx for idx, element in enumerate(elements, start=1)}


def encode_data(df, encoding_dict):
    """
    Replaces values in the DataFrame with their corresponding codes.
    Arguments:
        df : DataFrame - The data to encode.
        encoding_dict : dict - The mapping of events to codes.
    Returns:
        DataFrame - The encoded DataFrame.
    """
    for column in df.columns:
        df[column] = df[column].replace(encoding_dict)
    return df


def export_data_to_spmf_txt(df, encoding_dict, output_path):
    """
    Exports encoded data to an SPMF-compatible text file.
    Arguments:
        df : DataFrame - The encoded data.
        encoding_dict : dict - The mapping of events to codes.
        output_path : str - The path to save the SPMF file.
    """
    # Convert DataFrame to a list of records
    records = df.apply(lambda row: [str(val) for val in row], axis=1).tolist()

    with open(output_path, 'w') as file:
        file.write("@CONVERTED_FROM_TEXT\n")

        # Write encoded item mapping
        for event, code in encoding_dict.items():
            file.write(f"@ITEM={code}={event}\n")

        # Write sequences
        for record in records:
            file.write(" ".join(record) + "\n")


def remove_false_items(file_path):
    """
    Removes "False" items from the SPMF file by updating both item definitions and sequences.
    Arguments:
        file_path : str - The path to the SPMF file to modify.
    """
    items_to_remove = []

    # Identify items labeled as "False"
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("@"):
                parts = line.split("=")
                if parts[-1] == "False":
                    items_to_remove.append(parts[1])

    # Remove "False" items from sequences
    with open(file_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if line.startswith("@"):
            updated_lines.append(line)
        else:
            items = line.split()
            updated_lines.append(" ".join(item for item in items if item not in items_to_remove))

    # Write the cleaned file
    with open(file_path, "w") as file:
        file.write("\n".join(updated_lines))
