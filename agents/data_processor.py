import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
import logging
import re
from collections import Counter

from utils.file_handler import FileHandler
from utils.data_cleaner import DataCleaner

class DataProcessorAgent:
    """
    Agent 1: Responsible for processing ticket data, data cleansing, 
    analysis, and presenting the data in a readable format.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.raw_data = None
        self.processed_data = None
        self.file_info = None
        self.column_map = None
        self.summary_stats = None
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process the uploaded ticket file.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing file: {file_path}")
        
        # Step 1: Load the file
        self.raw_data, self.file_info = FileHandler.load_file(file_path)
        self.logger.info(f"Loaded file with {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
        
        # Step 2: Identify column types
        self.column_map = FileHandler.identify_column_types(self.raw_data)
        
        # Step 3: Standardize column names
        standardized_df = FileHandler.standardize_column_names(self.raw_data)
        self.logger.info("Standardized column names")
        
        # Step 4: Basic data cleaning
        cleaned_df = DataCleaner.clean_dataframe(standardized_df)
        self.logger.info("Performed basic data cleaning")
        
        # Step 5: Convert date columns
        date_columns = self.column_map["date_columns"]
        date_df = DataCleaner.convert_date_columns(cleaned_df, date_columns)
        self.logger.info("Converted date columns to datetime format")
        
        # Step 6: Calculate resolution time if not present
        resolution_df = DataCleaner.calculate_resolution_time(date_df)
        self.logger.info("Calculated or standardized resolution time")
        
        # Step 7: Clean text data
        text_columns = self.column_map["description_columns"] + self.column_map["resolution_columns"]
        text_cleaned_df = DataCleaner.clean_text_data(resolution_df, text_columns)
        self.logger.info("Cleaned text columns")
        
        # Step 8: Add derived features
        self.processed_data = DataCleaner.add_derived_features(text_cleaned_df)
        self.logger.info("Added derived features")
        
        # Step 9: Generate summary statistics
        self.summary_stats = self._generate_summary_statistics()
        self.logger.info("Generated summary statistics")
        
        return {
            "file_info": self.file_info,
            "column_map": self.column_map,
            "raw_data": self.raw_data,
            "processed_data": self.processed_data,
            "summary_stats": self.summary_stats
        }
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the processed data.
        
        Returns:
            Dictionary containing summary statistics
        """
        stats = {}
        df = self.processed_data
        
        # Basic counts
        stats["total_tickets"] = len(df)
        
        # Time-based statistics if relevant columns exist
        if 'resolution_time' in df.columns and pd.api.types.is_numeric_dtype(df['resolution_time']):
            stats["avg_resolution_time"] = df['resolution_time'].mean()
            stats["median_resolution_time"] = df['resolution_time'].median()
            stats["min_resolution_time"] = df['resolution_time'].min()
            stats["max_resolution_time"] = df['resolution_time'].max()
            
            # Resolution time distribution
            if 'resolution_band' in df.columns:
                stats["resolution_band_distribution"] = df['resolution_band'].value_counts().to_dict()
        
        # Assignment group distribution if it exists
        if 'assignment_group' in df.columns:
            stats["assignment_group_distribution"] = df['assignment_group'].value_counts().head(10).to_dict()
        
        # Complexity distribution if it exists
        if 'complexity' in df.columns:
            stats["complexity_distribution"] = df['complexity'].value_counts().to_dict()
        
        # Time patterns if date columns exist
        if 'open_day_of_week' in df.columns:
            stats["day_of_week_distribution"] = df['open_day_of_week'].value_counts().to_dict()
        
        if 'open_month' in df.columns:
            stats["month_distribution"] = df['open_month'].value_counts().to_dict()
        
        if 'open_hour' in df.columns:
            stats["hour_distribution"] = df['open_hour'].value_counts().to_dict()
        
        # Text analysis if description column exists
        if 'description' in df.columns:
            # Extract common keywords
            all_text = ' '.join(df['description'].fillna('').astype(str))
            # Simple tokenization and filtering for keywords
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            # Remove common stop words (simplified approach)
            stop_words = {'and', 'the', 'to', 'of', 'is', 'in', 'it', 'for', 'with', 'on', 'at', 'this', 'that', 'was', 'not'}
            filtered_words = [word for word in words if word not in stop_words]
            # Get most common words
            common_words = Counter(filtered_words).most_common(20)
            stats["common_keywords"] = dict(common_words)
        
        # Similar analysis for close notes if they exist
        if 'close_notes' in df.columns:
            all_notes = ' '.join(df['close_notes'].fillna('').astype(str))
            note_words = re.findall(r'\b[a-zA-Z]{3,}\b', all_notes.lower())
            filtered_note_words = [word for word in note_words if word not in stop_words]
            common_note_words = Counter(filtered_note_words).most_common(20)
            stats["common_resolution_keywords"] = dict(common_note_words)
        
        return stats
    
    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Generate visualizations for the processed data.
        
        Returns:
            Dictionary containing visualization figures
        """
        if self.processed_data is None:
            self.logger.error("No processed data available for visualization")
            return {}
        
        visualizations = {}
        df = self.processed_data
        
        # 1. Resolution time distribution
        if 'resolution_time' in df.columns and pd.api.types.is_numeric_dtype(df['resolution_time']):
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(df['resolution_time'].clip(upper=df['resolution_time'].quantile(0.95)), bins=20)
            plt.title('Resolution Time Distribution (95th Percentile)')
            plt.xlabel('Resolution Time (hours)')
            plt.ylabel('Frequency')
            visualizations["resolution_time_dist"] = fig
            
            # Resolution time by assignment group if both columns exist
            if 'assignment_group' in df.columns:
                fig = plt.figure(figsize=(12, 8))
                top_groups = df['assignment_group'].value_counts().head(10).index
                group_df = df[df['assignment_group'].isin(top_groups)]
                sns.boxplot(x='assignment_group', y='resolution_time', data=group_df)
                plt.title('Resolution Time by Assignment Group')
                plt.xticks(rotation=45)
                plt.xlabel('Assignment Group')
                plt.ylabel('Resolution Time (hours)')
                visualizations["resolution_by_group"] = fig
        
        # 2. Tickets by day of week if column exists
        if 'open_day_of_week' in df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['open_day_of_week'].value_counts().reindex(day_order).fillna(0)
            
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x=day_counts.index, y=day_counts.values)
            plt.title('Ticket Creation by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Tickets')
            visualizations["tickets_by_day"] = fig
        
        # 3. Ticket complexity distribution if column exists
        if 'complexity' in df.columns:
            complexity_order = ['Simple', 'Medium', 'Complex', 'Very Complex']
            complexity_counts = df['complexity'].value_counts().reindex(complexity_order).fillna(0)
            
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x=complexity_counts.index, y=complexity_counts.values)
            plt.title('Ticket Complexity Distribution')
            plt.xlabel('Complexity')
            plt.ylabel('Number of Tickets')
            visualizations["complexity_dist"] = fig
        
        # 4. Bar chart of common words for descriptions if column exists
        if 'description' in df.columns:
            all_text = ' '.join(df['description'].fillna('').astype(str))
            # Simple stopwords filtering
            stop_words = {'and', 'the', 'to', 'of', 'is', 'in', 'it', 'for', 'with', 'on', 'at', 'this', 'that', 'was', 'not'}
            
            # Extract words and count frequencies
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            filtered_words = [word for word in words if word not in stop_words]
            word_counts = pd.Series(filtered_words).value_counts().head(20)
            
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(x=word_counts.values, y=word_counts.index)
            plt.title('Common Words in Ticket Descriptions')
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            visualizations["description_word_freq"] = fig
        
        # 5. Bar chart of common words for resolution notes if column exists
        if 'close_notes' in df.columns:
            all_notes = ' '.join(df['close_notes'].fillna('').astype(str))
            
            # Extract words and count frequencies
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_notes.lower())
            filtered_words = [word for word in words if word not in stop_words]
            word_counts = pd.Series(filtered_words).value_counts().head(20)
            
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(x=word_counts.values, y=word_counts.index)
            plt.title('Common Words in Resolution Notes')
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            visualizations["resolution_word_freq"] = fig
        
        return visualizations
    
    def prepare_data_for_agent2(self) -> Dict[str, Any]:
        """
        Prepare a data package for Agent 2.
        
        Returns:
            Dictionary containing data for Agent 2 analysis
        """
        if self.processed_data is None:
            self.logger.error("No processed data available for Agent 2")
            return {}
        
        # Create a focused dataset for Agent 2
        agent2_data = {
            "processed_data": self.processed_data,
            "summary_stats": self.summary_stats,
            "column_map": self.column_map,
            "text_columns": {}
        }
        
        # Extract text data for pattern analysis
        df = self.processed_data
        
        # Add resolution notes if available
        if 'close_notes' in df.columns:
            agent2_data["text_columns"]["close_notes"] = df['close_notes'].fillna('').astype(str).tolist()
            
        # Add description text if available
        if 'description' in df.columns:
            agent2_data["text_columns"]["description"] = df['description'].fillna('').astype(str).tolist()
            
        # Add resolution if available
        if 'resolution' in df.columns:
            agent2_data["text_columns"]["resolution"] = df['resolution'].fillna('').astype(str).tolist()
            
        # Include ticket IDs for reference
        id_column = self.column_map["id_columns"][0] if self.column_map["id_columns"] else None
        if id_column and id_column in df.columns:
            agent2_data["ticket_ids"] = df[id_column].tolist()
        
        return agent2_data