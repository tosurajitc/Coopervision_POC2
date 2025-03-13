import pandas as pd
import numpy as np
import re
from datetime import datetime
from rate_limit_handler import apply_rate_limit_handling

@apply_rate_limit_handling
class DataProcessingAgent:
    """
    Agent 1: Data Processing & Presentation
    - Processes ticket data
    - Cleanses and structures the data
    - Prepares data for analysis
    """
    
    def __init__(self):
        self.required_columns = [
            'ticket_id', 'description', 'resolution', 'assignment_group'
        ]
    
    def process_data(self, data):
        """
        Process the uploaded ticket data
        
        Args:
            data (pd.DataFrame): The raw ticket data
            
        Returns:
            pd.DataFrame: Processed data ready for analysis
        """
        # Make a copy of the data to avoid modifying the original
        processed_data = data.copy()
        
        # Standardize column names (convert to lowercase and replace spaces with underscores)
        processed_data.columns = [self._normalize_column_name(col) for col in processed_data.columns]
        
        # Check for required columns and map to standard names if necessary
        processed_data = self._ensure_required_columns(processed_data)
        
        # Clean text fields
        text_columns = ['description', 'resolution', 'closed_notes', 'comments']
        for col in text_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].apply(
                    lambda x: self._clean_text(x) if pd.notna(x) else '')
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Extract additional features
        processed_data = self._extract_features(processed_data)
        
        return processed_data
    
    def _normalize_column_name(self, column_name):
        """Normalize column names to a standard format"""
        # Convert to lowercase and replace spaces with underscores
        normalized = str(column_name).lower().replace(' ', '_')
        # Remove special characters
        normalized = re.sub(r'[^\w_]', '', normalized)
        return normalized
    
    def _ensure_required_columns(self, data):
        """
        Ensure required columns exist, attempt to map similar columns if needed
        """
        column_mapping = {
            'ticket_id': ['id', 'incident_id', 'ticket_number', 'case_id', 'incident_number'],
            'description': ['desc', 'issue', 'issue_description', 'problem_description', 'short_description', 'summary'],
            'resolution': ['resolution_notes', 'resolve_notes', 'solution', 'fix', 'resolution_description', 'closed_notes'],
            'assignment_group': ['assigned_group', 'support_group', 'support_team', 'team', 'assigned_to_group']
        }
        
        # Try to map columns
        for required, alternatives in column_mapping.items():
            if required not in data.columns:
                # Find alternative column names
                for alt in alternatives:
                    if alt in data.columns:
                        data[required] = data[alt]
                        break
                        
                # If still not found, create empty column
                if required not in data.columns:
                    data[required] = np.nan
        
        return data
    
    def _clean_text(self, text):
        """Clean text data"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:?!-]', '', text)
        
        return text.strip()
    
    def _handle_missing_values(self, data):
        """Handle missing values in the data"""
        # Fill missing text values with empty string
        text_columns = ['description', 'resolution', 'closed_notes', 'comments']
        for col in text_columns:
            if col in data.columns:
                data[col] = data[col].fillna('')
        
        # Fill missing categorical values with 'Unknown'
        categorical_columns = ['assignment_group', 'status', 'priority', 'category']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
        
        # Fill missing numeric values with 0
        numeric_columns = ['duration', 'age', 'reassignment_count']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        return data
    
    def _extract_features(self, data):
        """Extract additional features for analysis"""
        # Extract ticket age/duration if date columns exist
        date_columns = [col for col in data.columns if 'date' in col or 'time' in col]
        
        if len(date_columns) >= 2:
            try:
                # Try to find created/opened and resolved/closed date columns
                created_col = next((col for col in date_columns if 'create' in col or 'open' in col), None)
                resolved_col = next((col for col in date_columns if 'resolve' in col or 'close' in col), None)
                
                if created_col and resolved_col:
                    # Convert to datetime
                    data[created_col] = pd.to_datetime(data[created_col], errors='coerce')
                    data[resolved_col] = pd.to_datetime(data[resolved_col], errors='coerce')
                    
                    # Calculate duration in hours
                    data['duration_hours'] = (data[resolved_col] - data[created_col]).dt.total_seconds() / 3600
                    data['duration_hours'] = data['duration_hours'].fillna(0).clip(lower=0)
            except Exception as e:
                # If date processing fails, continue without this feature
                print(f"Error processing date columns: {str(e)}")
        
        # Extract keyword indicators based on common automation terms
        automation_keywords = [
            'manual', 'repetitive', 'routine', 'recurring', 'regular', 
            'time-consuming', 'tedious', 'daily', 'weekly', 'monthly',
            'data entry', 'copy paste', 'spreadsheet', 'report', 'upload',
            'download', 'password reset', 'account unlock', 'permission',
            'access request', 'onboarding', 'offboarding'
        ]
        
        # Check description and resolution for automation keywords
        if 'description' in data.columns:
            data['automation_keyword_in_desc'] = data['description'].apply(
                lambda x: any(keyword in x for keyword in automation_keywords)
            )
        
        if 'resolution' in data.columns:
            data['automation_keyword_in_res'] = data['resolution'].apply(
                lambda x: any(keyword in x for keyword in automation_keywords)
            )
        
        # Extract reassignment count if available
        reassignment_indicators = ['reassign', 'transfer', 'escalate', 'handoff', 'hand off', 'hand-off']
        if 'comments' in data.columns:
            data['reassignment_indicator'] = data['comments'].apply(
                lambda x: any(indicator in x for indicator in reassignment_indicators)
            )
        
        return data