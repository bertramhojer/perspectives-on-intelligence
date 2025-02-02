from typing import List

import numpy as np
import pandas as pd
from data_mappings import (criteria_mapping, entity_intelligence_mapping,
                           intelligence_test_mapping, lacking_criteria_mapping,
                           occupation_map, research_goals_mapping,
                           research_mapping)


def extract_choice_mapping(df, prefix):
    """
    Extract rows with a specific variable name prefix and create a mapping of choice values to choice texts.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with the survey structure
    prefix (str): Prefix to search for in the variableName column
    
    Returns:
    dict: Mapping of choice values to choice texts
    """
    # Filter rows that match the prefix
    mask = df['label'] == prefix
    filtered_df = df[mask]
    
    # Create mapping dictionary
    # Convert choiceValue to string to handle potential numeric values
    mapping = {row['choiceValue']: row['choiceText'] for _, row in filtered_df.iterrows()}
    
    return mapping


def get_selected_items(row, column_prefix, value_mapping, target_value=1):
    """
    Extract items that have a specified value from a given row of data.
    
    Parameters:
    row (pandas.Series): A row from the DataFrame containing the columns
    column_prefix (str): Prefix of the columns to check (e.g., 's_8_')
    value_mapping (dict): Dictionary mapping column names to readable names
    target_value (any, optional): Value to look for in the columns (default: 1)
    
    Returns:
    list: Names of items that were selected (had target_value)
    """
    selected_items = []
    
    # Iterate through the columns that start with the specified prefix
    for col in [c for c in row.index if c.startswith(column_prefix)]:
        # Check if the value matches the target value (handles float equivalents)
        if row[col] == target_value or row[col] == float(target_value):
            # Append the readable name if it exists in the mapping
            if col in value_mapping:
                selected_items.append(value_mapping[col])
            
    return selected_items


def create_dummy_variables(df, columns, prefix_map=None, drop_original=False):
    """
    Convert multiple DataFrame columns containing lists into dummy variables.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    columns (list or str): Column name(s) containing lists to convert
    prefix_map (dict, optional): Dictionary mapping column names to prefixes
                               If None, uses original column names
    drop_original (bool): Whether to drop the original columns in the output
    
    Returns:
    pandas.DataFrame: New DataFrame with dummy variables added
    """
    # Create a copy of the original DataFrame
    result_df = df.copy()
    
    # Convert columns parameter to list if it's a string
    if isinstance(columns, str):
        columns = [columns]
    
    # Initialize prefix_map if None
    if prefix_map is None:
        prefix_map = {col: col for col in columns}
    else:
        # Ensure all columns have a prefix
        for col in columns:
            if col not in prefix_map:
                prefix_map[col] = col
    
    # Process each column
    for column in columns:
        # Get all unique values from all lists in the column
        unique_items = set()
        for item_list in df[column]:
            if isinstance(item_list, list):
                unique_items.update(item_list)
        unique_items = sorted(unique_items)
        
        # Create dummy variables for this column
        prefix = prefix_map[column]
        for item in unique_items:
            # Create column name
            col_name = f"{prefix}_{item.replace(' ', '_').replace('/', '_')}"
            
            # Create the dummy variable
            result_df[col_name] = df[column].apply(
                lambda x: 1 if isinstance(x, list) and item in x else 0
            )
    
    # Drop original columns if requested
    if drop_original:
        result_df = result_df.drop(columns=columns)
    
    return result_df

# Define mapping configurations
mapping_configs = [
    ('supp_research_area', 's_8_', research_mapping),
    ('occupation', 's_3_', occupation_map),
    ('criteria', 's_23_', criteria_mapping),
    ('lacking_criteria', 's_25_', lacking_criteria_mapping),
    ('entity_intelligence', 's_17_', entity_intelligence_mapping),
    ('intelligence_tests', 's_19_', intelligence_test_mapping),
    ('research_goals', 's_27_', research_goals_mapping)
]


columns_to_encode = [
    'supp_research_area',
    'criteria',
    'lacking_criteria',
    'entity_intelligence',
    'intelligence_tests',
    'research_goals',
    'occupation'
]

prefix_map = {
    'supp_research_area': 'supp',
    'criteria': 'crit',
    'lacking_criteria': 'lack',
    'entity_intelligence': 'ent',
    'intelligence_tests': 'test',
    'research_goals': 'goal',
    'occupation': 'occ'
}


def get_processed_datasets():

    # Load data
    labels = pd.read_csv("data/raw/labels.csv", sep=';', encoding="ISO-8859-1", names=["label", "choiceValue", "choiceText"])
    data = pd.read_csv("data/raw/dataset.csv", sep=';', encoding="ISO-8859-1")
    print("Full dataset: ", len(data))
    data = data[data['statoverall_4'] == 1].reset_index(drop=True)
    print("Filtered dataset: ", len(data))

    # Create single mapping
    single_mapping = {
        "career_stage": pd.Series([data["s_2"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_2")),
        "primary_area": pd.Series([data["s_7"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_7")),
        "origin_region": pd.Series([data["s_4"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_4")),
        "work_region": pd.Series([data["s_5"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_5")),
        "gender": pd.Series([data["s_6"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_6")),
        "other_criteria": pd.Series([data["s_16"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_16")),
        "llm_intelligence": pd.Series([data["s_9"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_9")),
        "llm_intelligence_future": pd.Series([data["s_26"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_26")),
        "agreement": pd.Series([data["s_18"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_18")),
        "comments": pd.Series([data["s_11"][i] for i in range(len(data))]).map(extract_choice_mapping(labels, "s_11")),
    }

    # Create a new dataframe with all mapped columns
    mapped_data = pd.DataFrame()
    for col_name, prefix, mapping in mapping_configs:
        mapped_data[col_name] = data.apply(
            lambda row: get_selected_items(row, prefix, mapping),
            axis=1
        )

    processed_data = pd.DataFrame(single_mapping)
    processed_data = pd.concat([processed_data, mapped_data], axis=1)

    processed_data = processed_data[processed_data['research_goals'].apply(lambda x: x != [])].reset_index(drop=True)

    dummy_data = create_dummy_variables(
        df=processed_data,
        columns=columns_to_encode,
        prefix_map=prefix_map,
        drop_original=True
    )

    return processed_data, dummy_data


def get_comment_dataset():
    """
    Extract free-text comment fields from the survey data and create a separate CSV file.
    
    Returns:
    pandas.DataFrame: DataFrame containing the comment fields
    """
    # Load raw data
    labels = pd.read_csv("data/raw/labels.csv", sep=';', encoding="ISO-8859-1", names=["label", "choiceValue", "choiceText"])
    data = pd.read_csv("data/raw/dataset.csv", sep=';', encoding="ISO-8859-1")
    data = data.dropna(subset=['s_1']).reset_index(drop=True)

    # Define comment field mappings (assuming these are the typical free-text fields in surveys)
    comment_columns = {
        's_11': 'research_goals',          # Based on your existing mapping
        's_24': 'criteria_comments',         # Comments about intelligence criteria
        's_28': 'primary_research_area_other',   # Comments about research goals
        's_29': 'secondary_research_area_other', # Comments about research goals
        's_16': 'other_criteria_comments', # Comments about other criteria
        's_10': 'intelligence_comments', # Comments about intelligence
        's_21': 'general_comments', # Comments about general
    }
    
    # Create DataFrame for comments
    comments_df = pd.DataFrame()
    
    # Extract each comment field
    for col_id, col_name in comment_columns.items():
        comments_df[col_name] = data[col_id]
    
    # Drop rows where all columns are NaN
    comments_df = comments_df.dropna(how='all')
    
    return comments_df


if __name__ == "__main__":

    processed_data, dummy_data = get_processed_datasets()
    comments_df = get_comment_dataset()

    print(processed_data.head())    
    print(dummy_data.head())

    processed_data.to_csv("data/processed/data.csv")
    dummy_data.to_csv("data/processed/data_encoded.csv")
    processed_data.to_csv("data/processed/data.csv")
    dummy_data.to_csv("data/processed/data_encoded.csv")
    dummy_data.to_csv("data/processed/data_encoded.csv")

    comments_df.to_csv("data/processed/comments.csv", index=False)
