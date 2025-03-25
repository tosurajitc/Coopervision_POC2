import pandas as pd
import re

def preprocess_ticket_data(data):
    """
    Preprocess the ticket data for visualization by normalizing column names
    and handling common data issues
    
    Args:
        data (pd.DataFrame): Raw or processed ticket data
        
    Returns:
        pd.DataFrame: Data with standardized column names
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Standardize column names
    df.columns = [normalize_column_name(col) for col in df.columns]
    
    # Handle External Ref# column specifically
    if 'external_ref#' in df.columns:
        df.rename(columns={'external_ref#': 'external_ref'}, inplace=True)
    
    # Convert date columns to datetime
    date_columns = [col for col in df.columns if col in ['opened', 'closed'] or 'date' in col]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass  # Skip if conversion fails
    
    # Ensure priority is standardized (if present)
    if 'priority' in df.columns:
        # Map various priority formats to standard format
        priority_mapping = {
            'p1': 'P1',
            'p2': 'P2',
            'p3': 'P3',
            'p4': 'P4',
            'p5': 'P5',
            'critical': 'P1',
            'high': 'P2',
            'medium': 'P3',
            'low': 'P4',
            'planning': 'P5'
        }
        
        # Convert priorities to string and lowercase for mapping
        df['priority'] = df['priority'].astype(str).str.lower()
        
        # Apply mapping for known values, otherwise keep as is
        df['priority'] = df['priority'].apply(
            lambda x: priority_mapping.get(x, x.capitalize() if x in priority_mapping.keys() else x.capitalize())
        )
    
    # Handle missing values in important columns
    categorical_columns = ['priority', 'assignment_group', 'subcategory', 'subcategory_2', 'subcategory_3']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Trim whitespace from string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
    
    return df

def normalize_column_name(column_name):
    """
    Normalize column names to a standard format
    
    Args:
        column_name (str): Original column name
        
    Returns:
        str: Normalized column name
    """
    # Convert to lowercase
    normalized = str(column_name).lower()
    
    # Replace spaces with underscores
    normalized = normalized.replace(' ', '_')
    
    # Remove special characters except for underscores
    normalized = re.sub(r'[^\w_]', '', normalized)
    
    # Return the normalized column name
    return normalized

def map_column_names(data):
    """
    Map column names from the expected columns provided in the prompt
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with mapped column names
    """
    # Column name mapping based on the expected columns from the prompt
    mapping = {
        'number': 'ticket_id',
        'priority': 'priority',
        'opened': 'opened',
        'assignment_group': 'assignment_group',
        'subcategory': 'subcategory',
        'subcategory_3': 'subcategory_3',
        'subcategory_2': 'subcategory_2',
        'short_description': 'short_description',
        'state': 'state',
        'opened_by': 'opened_by',
        'assigned_to': 'assigned_to',
        'closed': 'closed',
        'work_notes': 'work_notes',
        'correlation_display': 'correlation_display',
        'external_ref': 'external_ref',
        'ebs_resolution': 'ebs_resolution',
        'ebs_resolution_tier_2': 'ebs_resolution_tier_2',
        'ebs_resolution_tier_3': 'ebs_resolution_tier_3',
        'close_notes': 'close_notes'
    }
    
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Apply the mapping to the columns that exist
    columns_to_rename = {col: mapping[col] for col in df.columns if col in mapping}
    df.rename(columns=columns_to_rename, inplace=True)
    
    return df