import pandas as pd
import os
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file loading and initial processing of ticket data files."""
    
    @staticmethod
    def load_file(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load ticket data from CSV or Excel file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple containing DataFrame and metadata about the file
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        file_info = {
            "filename": os.path.basename(file_path),
            "extension": file_ext,
            "size_mb": os.path.getsize(file_path) / (1024 * 1024)
        }
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                file_info["rows"] = len(df)
                file_info["columns"] = list(df.columns)
                logger.info(f"Successfully loaded CSV file with {len(df)} rows")
                
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
                file_info["rows"] = len(df)
                file_info["columns"] = list(df.columns)
                logger.info(f"Successfully loaded Excel file with {len(df)} rows")
                
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                raise ValueError(f"Unsupported file format: {file_ext}. Please upload a .csv, .xls, or .xlsx file.")
                
            return df, file_info
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    @staticmethod
    def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify and categorize columns in the ticket data.
        
        Args:
            df: DataFrame containing ticket data
            
        Returns:
            Dictionary mapping column types to column names
        """
        columns = list(df.columns)
        column_map = {
            "id_columns": [],
            "date_columns": [],
            "description_columns": [],
            "resolution_columns": [],
            "time_columns": [],
            "assignment_columns": [],
            "other_columns": []
        }
        
        # Simple heuristic-based approach to categorize columns
        for col in columns:
            col_lower = col.lower()
            
            # ID columns
            if any(term in col_lower for term in ['id', 'number', 'ticket', 'case']):
                column_map["id_columns"].append(col)
                
            # Date columns
            elif any(term in col_lower for term in ['date', 'created', 'opened', 'closed', 'resolved']):
                column_map["date_columns"].append(col)
                
            # Description columns
            elif any(term in col_lower for term in ['desc', 'description', 'summary', 'title', 'issue']):
                column_map["description_columns"].append(col)
                
            # Resolution columns
            elif any(term in col_lower for term in ['resolution', 'resolve', 'solution', 'notes', 'comment']):
                column_map["resolution_columns"].append(col)
                
            # Time columns
            elif any(term in col_lower for term in ['time', 'duration', 'sla']):
                column_map["time_columns"].append(col)
                
            # Assignment columns
            elif any(term in col_lower for term in ['assign', 'group', 'team', 'owner', 'responsible']):
                column_map["assignment_columns"].append(col)
                
            # Other columns
            else:
                column_map["other_columns"].append(col)
        
        return column_map
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a standardized version of the DataFrame with consistent column naming.
        
        Args:
            df: Original DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        column_map = FileHandler.identify_column_types(df)
        standardized_df = df.copy()
        
        # Create a mapping of original columns to standardized column names
        # This helps create a more consistent interface for downstream agents
        rename_map = {}
        
        # Handle ID columns
        if column_map["id_columns"]:
            rename_map[column_map["id_columns"][0]] = "ticket_id"
        
        # Handle description columns
        if column_map["description_columns"]:
            rename_map[column_map["description_columns"][0]] = "description"
        
        # Handle date columns
        date_cols = column_map["date_columns"]
        if date_cols:
            open_date_cols = [col for col in date_cols if any(term in col.lower() for term in ['open', 'create', 'start'])]
            close_date_cols = [col for col in date_cols if any(term in col.lower() for term in ['close', 'resolv', 'end'])]
            
            if open_date_cols:
                rename_map[open_date_cols[0]] = "open_date"
            if close_date_cols:
                rename_map[close_date_cols[0]] = "close_date"
        
        # Handle resolution columns
        if column_map["resolution_columns"]:
            resolution_cols = [col for col in column_map["resolution_columns"] if 'resolution' in col.lower()]
            notes_cols = [col for col in column_map["resolution_columns"] if 'note' in col.lower()]
            
            if resolution_cols:
                rename_map[resolution_cols[0]] = "resolution"
            if notes_cols:
                rename_map[notes_cols[0]] = "close_notes"
        
        # Handle time columns
        if column_map["time_columns"]:
            rename_map[column_map["time_columns"][0]] = "resolution_time"
        
        # Handle assignment columns
        if column_map["assignment_columns"]:
            rename_map[column_map["assignment_columns"][0]] = "assignment_group"
        
        # Rename columns
        standardized_df = standardized_df.rename(columns=rename_map)
        
        return standardized_df