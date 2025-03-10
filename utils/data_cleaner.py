import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCleaner:
    """Provides utilities for cleaning and preprocessing ticket data."""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic cleaning operations on the ticket dataframe.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Drop duplicate rows if any
        initial_rows = len(clean_df)
        clean_df = clean_df.drop_duplicates()
        if len(clean_df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(clean_df)} duplicate rows")
        
        # Remove completely empty rows
        clean_df = clean_df.dropna(how='all')
        
        # Clean string columns - strip whitespace and handle special characters
        for col in clean_df.columns:
            if clean_df[col].dtype == 'object':
                clean_df[col] = clean_df[col].astype(str).str.strip()
                
                # Replace newlines and special chars in string columns
                clean_df[col] = clean_df[col].str.replace('\n', ' ', regex=False)
                clean_df[col] = clean_df[col].str.replace('\r', ' ', regex=False)
                clean_df[col] = clean_df[col].str.replace('\t', ' ', regex=False)
                clean_df[col] = clean_df[col].str.replace('  ', ' ', regex=False)
        
        return clean_df
    
    @staticmethod
    def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Convert date columns to datetime format.
        
        Args:
            df: Input DataFrame
            date_columns: List of column names containing date information
            
        Returns:
            DataFrame with converted date columns
        """
        result_df = df.copy()
        
        for col in date_columns:
            if col in result_df.columns:
                try:
                    # Try pandas auto-detection first
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                    
                    # Check if we have too many NaT values after conversion
                    na_percent = result_df[col].isna().mean() * 100
                    if na_percent > 30:  # More than 30% NaT values
                        logger.warning(f"High NaT rate ({na_percent:.1f}%) in column {col} after date conversion")
                        
                    logger.info(f"Successfully converted column {col} to datetime")
                except Exception as e:
                    logger.error(f"Failed to convert column {col} to datetime: {str(e)}")
        
        return result_df
    
    @staticmethod
    def calculate_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate or standardize resolution time if it's not already present.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with resolution_time column
        """
        result_df = df.copy()
        
        # Check if open_date and close_date columns exist
        if 'open_date' in result_df.columns and 'close_date' in result_df.columns:
            # Ensure they are datetime type
            if pd.api.types.is_datetime64_any_dtype(result_df['open_date']) and pd.api.types.is_datetime64_any_dtype(result_df['close_date']):
                # Calculate resolution time in hours
                result_df['calculated_resolution_time'] = (result_df['close_date'] - result_df['open_date']).dt.total_seconds() / 3600
                
                # If resolution_time exists, compare with calculated
                if 'resolution_time' in result_df.columns:
                    # Check if they are significantly different
                    logger.info("Comparing provided resolution_time with calculated values")
                else:
                    # Use calculated as resolution_time
                    result_df['resolution_time'] = result_df['calculated_resolution_time']
                    logger.info("Added calculated resolution_time column")
                    
                # Drop the temporary column
                result_df = result_df.drop('calculated_resolution_time', axis=1)
        
        return result_df
    
    @staticmethod
    def clean_text_data(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Clean and standardize text columns for better analysis.
        
        Args:
            df: Input DataFrame
            text_columns: List of columns containing textual data
            
        Returns:
            DataFrame with cleaned text columns
        """
        result_df = df.copy()
        
        for col in text_columns:
            if col in result_df.columns:
                # Convert to string type if not already
                result_df[col] = result_df[col].astype(str)
                
                # Remove URLs
                result_df[col] = result_df[col].str.replace(r'http\S+', '', regex=True)
                
                # Remove special characters but keep spaces
                result_df[col] = result_df[col].str.replace(r'[^\w\s]', ' ', regex=True)
                
                # Remove extra whitespace
                result_df[col] = result_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                
                # Replace nan or None values with empty string
                result_df[col] = result_df[col].replace(['nan', 'None', 'NaN'], '')
        
        return result_df
    
    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful derived features for analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        result_df = df.copy()
        
        # Add day of week for open_date if it exists
        if 'open_date' in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df['open_date']):
            result_df['open_day_of_week'] = result_df['open_date'].dt.day_name()
            result_df['open_hour'] = result_df['open_date'].dt.hour
            result_df['open_month'] = result_df['open_date'].dt.month_name()
        
        # Calculate complexity from description length if description exists
        if 'description' in result_df.columns:
            # Word count as a simple complexity measure
            result_df['description_word_count'] = result_df['description'].astype(str).str.split().str.len()
            
            # Categorize complexity
            result_df['complexity'] = pd.cut(
                result_df['description_word_count'],
                bins=[0, 20, 50, 100, float('inf')],
                labels=['Simple', 'Medium', 'Complex', 'Very Complex']
            )
        
        # Add processing time bands if resolution_time exists
        if 'resolution_time' in result_df.columns:
            # Convert to numeric if not already
            if not pd.api.types.is_numeric_dtype(result_df['resolution_time']):
                result_df['resolution_time'] = pd.to_numeric(result_df['resolution_time'], errors='coerce')
            
            # Create time bands (in hours)
            result_df['resolution_band'] = pd.cut(
                result_df['resolution_time'],
                bins=[0, 1, 4, 24, 72, float('inf')],
                labels=['< 1hr', '1-4hrs', '4-24hrs', '1-3 days', '> 3 days']
            )
        
        return result_df