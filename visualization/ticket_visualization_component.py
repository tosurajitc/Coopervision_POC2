import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

class TicketVisualizationComponent:
    """
    Component for creating ticket data visualizations
    - Creates various charts and visualizations
    - Analyzes External Ref# patterns
    - Integrates with Streamlit UI
    """
    
    def __init__(self):
        # Define consistent figure sizes for all charts
        self.figure_width = 6
        self.figure_height = 4
        
        # Define consistent colors
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'alert': '#d62728',
            'neutral': '#7f7f7f'
        }
        
        # Define a palette of additional colors
        self.palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Define region-specific colors
        self.region_colors = {
            'AMER': '#4285F4',   # Google Blue
            'APAC': '#EA4335',   # Google Red
            'EMEA': '#FBBC05',   # Google Yellow
            'GLOBAL': '#34A853', # Google Green
            'Unknown': '#9AA0A6'  # Google Grey
        }
        
        # Set consistent style for all plots
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up consistent plot style for all visualizations"""
        # Set the style
        plt.style.use('ggplot')
        
        # Set consistent font sizes across all plots
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
        
        # Increase DPI for sharper rendering
        plt.rcParams['figure.dpi'] = 100
        
        # Set figure size defaults
        plt.rcParams['figure.figsize'] = (self.figure_width, self.figure_height)
        
        # Improve aesthetics
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
    
    def create_visualizations(self, data):
        """
        Main method to create all visualizations
        
        Args:
            data (pd.DataFrame): Processed ticket data
            
        Returns:
            None (displays visualizations in Streamlit)
        """
        # Check if data is valid
        if data is None or len(data) == 0:
            st.warning("No data available for visualization.")
            return
        
        # Normalize column names to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]
        
        # Create visualization sections
        st.markdown("### 📊 Ticket Data Visualizations")
        
        # Create tabs for different visualization categories
        tab1, tab2, tab3 = st.tabs(["Distribution Charts", "Time Analysis", "External Ref Analysis"])
        
        with tab1:
            self._create_distribution_charts(data)
            
        with tab2:
            self._create_time_analysis(data)
            
        with tab3:
            self._analyze_external_ref(data)
    
    def _create_distribution_charts(self, data):
        """
        Create distribution charts including pie and bar charts
        """
        st.markdown("#### Ticket Distribution Analysis")
        
        # Create a two-column layout for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Priority Distribution (Pie Chart)
            if 'priority' in data.columns:
                self._create_pie_chart(
                    data, 
                    'priority', 
                    title="Ticket Distribution by Priority"
                )
            else:
                st.info("Priority column not found in the data.")
        
        with col2:
            # 2. Subcategory Distribution (Pie Chart)
            if 'subcategory' in data.columns:
                self._create_pie_chart(
                    data, 
                    'subcategory', 
                    title="Ticket Distribution by Subcategory",
                    limit=8
                )
            else:
                st.info("Subcategory column not found in the data.")
        
        # Create a horizontal rule for visual separation
        st.markdown("---")
        
        # Create a two-column layout for the additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            # 3. Subcategory 3 Distribution by Region (Pie Chart)
            if 'subcategory_3' in data.columns:
                # Extract region information
                region_data = self._extract_region_data(data)
                
                # Create pie chart for regions
                self._create_pie_chart(
                    region_data,
                    'region',
                    title="Ticket Distribution by Region",
                    use_region_colors=True
                )
            else:
                st.info("Subcategory 3 column not found in the data.")
        
        with col2:
            # 4. Subcategory 2 Distribution (Bar Chart)
            if 'subcategory_2' in data.columns:
                self._create_bar_chart(
                    data,
                    'subcategory_2',
                    title="Ticket Count by Subcategory 2",
                    limit=10,
                    color=self.colors['primary']
                )
            else:
                st.info("Subcategory 2 column not found in the data.")
        
        # Create a horizontal rule for visual separation
        st.markdown("---")
        
        # Create a two-column layout for EBS charts
        col1, col2 = st.columns(2)
        
        with col1:
            # 5. EBS Resolution Distribution (Bar Chart)
            if 'ebs_resolution' in data.columns:
                self._create_bar_chart(
                    data,
                    'ebs_resolution',
                    title="Ticket Count by EBS Resolution",
                    limit=8,
                    color=self.colors['tertiary']
                )
            else:
                st.info("EBS Resolution column not found in the data.")
        
        with col2:
            # 6. EBS Resolution Tier 2 Distribution (Bar Chart)
            if 'ebs_resolution_tier_2' in data.columns:
                self._create_bar_chart(
                    data,
                    'ebs_resolution_tier_2',
                    title="Ticket Count by EBS Resolution Tier 2",
                    limit=8,
                    color=self.colors['secondary']
                )
            else:
                st.info("EBS Resolution Tier 2 column not found in the data.")
    
    def _create_time_analysis(self, data):
        """
        Create time-based analysis
        """
        st.markdown("#### Time-Based Analysis")
        
        # Check if we have the necessary date columns
        date_columns = [col for col in data.columns if 'date' in col or col in ['opened', 'closed']]
        
        if len(date_columns) == 0:
            st.info("No date columns found for time analysis.")
            return
        
        # Ensure date columns are datetime type
        for col in date_columns:
            if col in data.columns:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                except:
                    st.warning(f"Could not convert {col} to datetime.")
        
        # Create a single column layout for the time series chart
        
        # Resolution Time Analysis
        if 'opened' in data.columns and 'closed' in data.columns:
            # Ensure both columns are datetime
            if pd.api.types.is_datetime64_dtype(data['opened']) and pd.api.types.is_datetime64_dtype(data['closed']):
                try:
                    # Calculate resolution time in hours
                    data['resolution_time_hours'] = (data['closed'] - data['opened']).dt.total_seconds() / 3600
                    
                    # Filter out negative or very large values that might indicate data issues
                    resolution_time_filtered = data[
                        (data['resolution_time_hours'] >= 0) & 
                        (data['resolution_time_hours'] < 500)  # 500 hours ~= 20 days
                    ]
                    
                    if len(resolution_time_filtered) > 0:
                        st.markdown("##### Resolution Time Distribution")
                        
                        # Create the histogram
                        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
                        sns.histplot(
                            resolution_time_filtered['resolution_time_hours'],
                            bins=20,
                            kde=True,
                            color=self.colors['secondary'],
                            ax=ax
                        )
                        
                        # Enhance the chart
                        plt.title("Resolution Time Distribution (Hours)")
                        plt.xlabel("Resolution Time (Hours)")
                        plt.ylabel("Number of Tickets")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # Display the chart
                        st.pyplot(fig)
                        
                        # Display resolution time statistics
                        st.markdown("##### Resolution Time Statistics")
                        stats = resolution_time_filtered['resolution_time_hours'].describe()
                        stats_df = pd.DataFrame({
                            'Statistic': ['Count', 'Mean (hours)', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                            'Value': [
                                f"{stats['count']:.0f}",
                                f"{stats['mean']:.2f}",
                                f"{stats['std']:.2f}",
                                f"{stats['min']:.2f}",
                                f"{stats['25%']:.2f}",
                                f"{stats['50%']:.2f}",
                                f"{stats['75%']:.2f}",
                                f"{stats['max']:.2f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error creating resolution time analysis: {e}")
    
    def _analyze_external_ref(self, data):
        """
        Analyze External Ref# patterns and create visualizations
        """
        st.markdown("#### External Reference Number Analysis")
        
        # Check if External Ref# column exists (handle variations in naming)
        ref_col = self._find_external_ref_column(data)
        
        if ref_col is None:
            st.info("External Ref# column not found in the data.")
            return
        
        # Filter out missing values
        ref_data = data[data[ref_col].notna()].copy()
        
        if len(ref_data) == 0:
            st.info("No External Ref# data available for analysis.")
            return
        
        # Extract patterns from External Ref#
        ref_data['ref_prefix'] = ref_data[ref_col].astype(str).apply(
            lambda x: re.search(r'^[A-Za-z]+', x).group(0) if re.search(r'^[A-Za-z]+', x) else 'Other'
        )
        
        # Extract reference length for analysis
        ref_data['ref_length'] = ref_data[ref_col].astype(str).apply(len)
        
        # Create a two-column layout for correlation analysis
        # and pattern detection
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation between External Ref# and other attributes
            if 'priority' in ref_data.columns:
                st.markdown("##### External Ref# by Priority")
                
                try:
                    # Create cross-tabulation
                    crosstab = pd.crosstab(ref_data['ref_prefix'], ref_data['priority'])
                    
                    # Plot stacked bar chart
                    fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
                    crosstab.plot(
                        kind='bar',
                        stacked=True,
                        ax=ax,
                        colormap='viridis'
                    )
                    
                    # Enhance the chart
                    plt.title("Distribution of External Ref# Prefixes by Priority")
                    plt.xlabel("External Ref# Prefix")
                    plt.ylabel("Number of Tickets")
                    plt.legend(title="Priority")
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    plt.tight_layout()
                    
                    # Display the chart
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Error creating External Ref# correlation analysis: {e}")
        
        with col2:
            # Pattern detection in External Ref#
            st.markdown("##### Common Patterns in External Ref#")
            
            try:
                # Extract the first part of the reference number (before any delimiter)
                pattern_analysis = pd.DataFrame({
                    'Full Ref': ref_data[ref_col].astype(str),
                    'First Part': ref_data[ref_col].astype(str).apply(
                        lambda x: x.split('-')[0] if '-' in x else (x.split('_')[0] if '_' in x else x)
                    )
                })
                
                # Count occurrences of first parts
                first_part_counts = pattern_analysis['First Part'].value_counts().head(10)
                
                # Plot bar chart
                fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
                first_part_counts.plot(
                    kind='bar',
                    color=self.colors['tertiary'],
                    ax=ax
                )
                
                # Enhance the chart
                plt.title("Most Common External Ref# Patterns")
                plt.xlabel("Pattern")
                plt.ylabel("Count")
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.tight_layout()
                
                # Display the chart
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Error creating External Ref# pattern analysis: {e}")
        
        # Display the most common full External Ref# values
        st.markdown("##### Most Common External Ref# Values")
        try:
            common_refs = ref_data[ref_col].value_counts().head(10).reset_index()
            common_refs.columns = ['External Ref#', 'Count']
            st.dataframe(common_refs, use_container_width=True)
        except Exception as e:
            st.warning(f"Error displaying common External Ref# values: {e}")
    
    def _extract_region_data(self, data):
        """
        Extract region information from data
        
        Args:
            data: DataFrame containing the data
        
        Returns:
            DataFrame with region column added
        """
        # Make a copy to avoid modifying the original
        region_data = data.copy()
        
        # Add region column with default value
        region_data['region'] = 'Unknown'
        
        # Try to extract region from different columns
        region_columns = ['subcategory_3', 'short_description', 'assignment_group']
        for col in region_columns:
            if col in data.columns:
                # Extract region only for rows where region is still Unknown
                mask = region_data['region'] == 'Unknown'
                
                # Use a pandas-friendly approach to avoid SettingWithCopyWarning
                if mask.any():
                    extracted_regions = region_data.loc[mask, col].apply(
                        lambda x: self._extract_region(x) if pd.notnull(x) else 'Unknown'
                    )
                    region_data.loc[mask, 'region'] = extracted_regions
        
        return region_data
    
    def _find_external_ref_column(self, data):
        """
        Find the External Ref# column in the dataframe
        
        Args:
            data: DataFrame to search in
            
        Returns:
            Column name or None if not found
        """
        possible_names = ['external_ref#', 'external_ref', 'externalref']
        
        # Check exact matches first
        for col in possible_names:
            if col in data.columns:
                return col
        
        # Check for partial matches
        for col in data.columns:
            if 'ref' in col.lower() and 'external' in col.lower():
                return col
        
        return None
    
    def _create_pie_chart(self, data, column, title="Distribution", limit=None, use_region_colors=False):
        """
        Create a pie chart for a categorical column
        
        Args:
            data: DataFrame containing the data
            column: Column name to analyze
            title: Chart title
            limit: Limit to top N categories
            use_region_colors: Whether to use region-specific colors
        """
        if column not in data.columns:
            st.info(f"Column {column} not found in the data.")
            return
        
        try:
            # Count the values, filter out missing data
            value_counts = data[column].value_counts()
            
            # Handle large number of categories
            if limit and len(value_counts) > limit:
                # Keep top N categories and group the rest as "Other"
                top_categories = value_counts.head(limit-1)
                other_count = value_counts.iloc[limit-1:].sum()
                
                # Add "Other" category
                value_counts = pd.Series({**top_categories.to_dict(), "Other": other_count})
            
            # Prepare figure and axes
            fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
            
            # Determine colors
            if use_region_colors and column == 'region':
                colors = [self.region_colors.get(region, self.colors['neutral']) for region in value_counts.index]
            else:
                colors = self.palette[:len(value_counts)]
            
            # Function to format percentage labels - only show for slices >= 5%
            def format_pct(pct):
                return f'{pct:.1f}%' if pct >= 5 else ''
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                value_counts,
                autopct=format_pct,
                pctdistance=0.85,
                startangle=90,
                colors=colors,
                textprops={'fontsize': 9, 'fontweight': 'bold'}
            )
            
            # Add legend
            ax.legend(
                wedges,
                value_counts.index,
                title=column.replace('_', ' ').title(),
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            # Set aspect ratio to be equal (circular pie)
            ax.set_aspect('equal')
            
            # Add title
            plt.title(title)
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            
            # Display top categories as a table
            top_n = min(5, len(value_counts))
            st.markdown(f"**Top {top_n} {column.replace('_', ' ').title()} Categories:**")
            
            table_data = pd.DataFrame({
                'Category': value_counts.index[:top_n],
                'Count': value_counts.values[:top_n],
                'Percentage': (value_counts.values[:top_n] / value_counts.sum() * 100).round(1)
            })
            table_data['Percentage'] = table_data['Percentage'].astype(str) + '%'
            
            st.dataframe(table_data, use_container_width=True, height=28*(top_n+1))
        except Exception as e:
            st.warning(f"Error creating pie chart for {column}: {e}")
    
    def _create_bar_chart(self, data, column, title="Count", limit=None, color=None):
        """
        Create a bar chart for a categorical column
        
        Args:
            data: DataFrame containing the data
            column: Column name to analyze
            title: Chart title
            limit: Limit to top N categories
            color: Bar color
        """
        if column not in data.columns:
            st.info(f"Column {column} not found in the data.")
            return
        
        try:
            # Count the values
            value_counts = data[column].value_counts()
            
            # Limit to top N categories
            if limit and len(value_counts) > limit:
                value_counts = value_counts.head(limit)
            
            # Prepare figure and axes
            fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
            
            # Create vertical bar chart
            bars = value_counts.plot(
                kind='bar',
                color=color if color else self.colors['primary'],
                ax=ax
            )
            
            # Add value labels to the bars
            for i, v in enumerate(value_counts):
                ax.text(
                    i,
                    v + (v * 0.02),  # Position labels just above bars
                    str(v),
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            # Set labels and title
            plt.title(title)
            plt.ylabel("Count")
            plt.xlabel(column.replace('_', ' ').title())
            
            # Add grid lines only on y-axis
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Improve layout
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            
            # Also show as a compact table
            table_data = pd.DataFrame({
                'Category': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / value_counts.sum() * 100).round(1)
            })
            table_data['Percentage'] = table_data['Percentage'].astype(str) + '%'
            
            st.dataframe(table_data, use_container_width=True, height=28*(len(value_counts)+1))
        except Exception as e:
            st.warning(f"Error creating bar chart for {column}: {e}")
    
    def _extract_region(self, text):
        """
        Extract region information from text
        
        Args:
            text: Text to extract region from
            
        Returns:
            Extracted region or 'Unknown'
        """
        if not isinstance(text, str):
            return 'Unknown'
        
        text = text.upper()
        
        if 'AMER' in text:
            return 'AMER'
        elif 'APAC' in text:
            return 'APAC'
        elif 'EMEA' in text:
            return 'EMEA'
        elif 'GLOBAL' in text:
            return 'GLOBAL'
        else:
            return 'Unknown'