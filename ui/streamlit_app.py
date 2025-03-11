import streamlit as st

# Set page config MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Ticket Automation Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
import logging
import time
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ensure the correct import paths
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_processor import DataProcessorAgent
from agents.insight_finder import InsightFinderAgent
from agents.user_query import UserQueryAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        min-width: 180px;
        flex: 1;
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4c85e5;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    div[data-baseweb="select"] span {
        max-width: 100%;
        white-space: normal;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: visible;
    }
    div[data-baseweb="select"] {
        min-height: 40px;
    }
    div[data-baseweb="popover"] div[role="listbox"] {
        max-width: 100%;
    }
    div[data-baseweb="popover"] div[role="option"] {
        white-space: normal;
        padding: 10px;
        line-height: 1.4;
    }
    /* Increase size of standard questions */
    .standard-questions-container label {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    .standard-questions-container div[data-baseweb="select"] span {
        font-size: 1.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'data_processor_results' not in st.session_state:
        st.session_state.data_processor_results = None
    
    if 'insight_finder_results' not in st.session_state:
        st.session_state.insight_finder_results = None
    
    if 'user_query_history' not in st.session_state:
        st.session_state.user_query_history = []
    
    if 'user_query_agent' not in st.session_state:
        st.session_state.user_query_agent = None
    
    if 'data_processor_agent' not in st.session_state:
        st.session_state.data_processor_agent = None
    
    if 'insight_finder_agent' not in st.session_state:
        st.session_state.insight_finder_agent = None

def process_uploaded_file(uploaded_file):
    """Process the uploaded ticket data file."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Show a progress bar for file processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize the Data Processor Agent if not already done
        if not st.session_state.data_processor_agent:
            st.session_state.data_processor_agent = DataProcessorAgent()
        
        # Update status
        status_text.text("Processing file... (1/3) Data Processing")
        progress_bar.progress(10)
        
        # Process the file with Agent 1
        data_processor_agent = st.session_state.data_processor_agent
        data_processor_results = data_processor_agent.process_file(tmp_path)
        
        # Update status
        status_text.text("Processing file... (2/3) Analyzing Patterns")
        progress_bar.progress(40)
        
        # Initialize and run Agent 2
        if not st.session_state.insight_finder_agent:
            st.session_state.insight_finder_agent = InsightFinderAgent()
        
        insight_finder_agent = st.session_state.insight_finder_agent
        data_for_agent2 = data_processor_agent.prepare_data_for_agent2()
        insight_finder_results = insight_finder_agent.process_data(data_for_agent2)
        
        # Update status
        status_text.text("Processing file... (3/3) Preparing Insights")
        progress_bar.progress(70)
        
        # Initialize Agent 3 for user queries
        if not st.session_state.user_query_agent:
            st.session_state.user_query_agent = UserQueryAgent()
        
        user_query_agent = st.session_state.user_query_agent
        user_query_agent.initialize_with_data(data_processor_results, insight_finder_results)
        
        # Update session state
        st.session_state.file_uploaded = True
        st.session_state.processed_data = data_processor_results.get("processed_data")
        st.session_state.data_processor_results = data_processor_results
        st.session_state.insight_finder_results = insight_finder_results
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("File processing complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"Error in process_uploaded_file: {str(e)}", exc_info=True)
        return False

def display_results():
    """Display the results of the data processing and analysis."""
    # Get the results from session state
    data_processor_results = st.session_state.data_processor_results
    insight_finder_results = st.session_state.insight_finder_results
    
    if not data_processor_results or not insight_finder_results:
        st.warning("No analysis results available. Please upload and process a file.")
        return
    
    # Create tabs for different views with consistent widths
    tab_labels = ["Data Overview", "Patterns", "Automation Suggestions", "💬 Custom Query"]
    tabs = st.tabs(tab_labels)
    
    # Tab 1: Data Overview
    with tabs[0]:
        # First section: Ask Questions To Get Some Insight
        st.header("🔍 Ask Questions To Get Some Insight")
        
        # Get standard questions from Agent 3
        if st.session_state.user_query_agent:
            # Standard questions dropdown
            standard_questions = st.session_state.user_query_agent.get_standard_questions()
            
            # Format questions for display with larger size (h2)
            st.markdown("## Standard Questions")
            st.markdown("Select a question to analyze your ticket data and get Qualititave Response:")
            
            # Create a dropdown with questions for selection
            question_options = ["Select a question..."] + [q["question"] for q in standard_questions]
            
            # Wrap in a div to apply custom CSS for larger size
            st.markdown('<div class="standard-questions-container">', unsafe_allow_html=True)
            selected_question = st.selectbox("", question_options, label_visibility="collapsed", key="data_overview_question")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if selected_question != "Select a question...":
                # Find the question ID and display the description
                for q in standard_questions:
                    if q["question"] == selected_question:
                        question_id = q["id"]
                        st.info(q["description"])
                        break
                
                if st.button("Get Answer", key="data_overview_answer_btn"):
                    with st.spinner("Analyzing..."):
                        response = st.session_state.user_query_agent.process_query(
                            selected_question, 
                            is_standard=True, 
                            question_id=question_id
                        )
                        st.session_state.user_query_history.append(response)
                        st.rerun()
        
        # If there are query results, display the latest on top
        if st.session_state.user_query_history:
            st.markdown("### Latest Response")
            latest_query = st.session_state.user_query_history[-1]
            
            # Display question in a colored box
            st.markdown(f"""
            <div style="background-color:#e6f0ff; padding:10px; border-radius:5px; margin-bottom:10px;">
                <strong>Question:</strong> {latest_query.get('question', 'Unknown query')}
            </div>
            """, unsafe_allow_html=True)
            
            # Clean up response text
            response_text = latest_query.get("response", "No response available.")
            if "<think>" in response_text:
                response_text = response_text.split("<think>")[0].strip()
            if response_text.startswith("Response:"):
                response_text = response_text[9:].strip()
            
            st.write(response_text)

        # Add a divider between sections
        st.markdown("---")
        
        # Second section: Ticket Data Overview
        st.header("Ticket Data Overview")
        
        # File information
        file_info = data_processor_results.get("file_info", {})
        if file_info:
            st.subheader("File Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Tickets", file_info.get("rows", 0))
            with col3:
                st.metric("File Size", f"{file_info.get('size_mb', 0):.2f} MB")
        
        # Summary statistics
        summary_stats = data_processor_results.get("summary_stats", {})
        if summary_stats:
            st.subheader("Summary Statistics")
            
            # Basic metrics in columns
            col1, col2, col3 = st.columns(3)
            
            # Resolution time metrics
            if "avg_resolution_time" in summary_stats:
                with col1:
                    st.metric("Avg. Resolution Time", f"{summary_stats['avg_resolution_time']:.2f} hrs")
            if "median_resolution_time" in summary_stats:
                with col2:
                    st.metric("Median Resolution Time", f"{summary_stats['median_resolution_time']:.2f} hrs")
            if "max_resolution_time" in summary_stats:
                with col3:
                    st.metric("Max Resolution Time", f"{summary_stats['max_resolution_time']:.2f} hrs")
            
            # Resolution band distribution
            if "resolution_band_distribution" in summary_stats:
                st.subheader("Resolution Time Distribution")
                band_data = pd.DataFrame({
                    "Time Band": list(summary_stats["resolution_band_distribution"].keys()),
                    "Count": list(summary_stats["resolution_band_distribution"].values())
                })
                
                fig = px.bar(band_data, x="Time Band", y="Count", title="Tickets by Resolution Time")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Patterns
    with tabs[1]:
        st.header("Identified Patterns")
        
        patterns = insight_finder_results.get("patterns", [])
        if not patterns:
            st.info("No significant patterns identified in the ticket data.")
        else:
            for i, pattern in enumerate(patterns):
                with st.expander(f"Pattern {i+1}: {', '.join(pattern.get('keywords', [])[:3])}"):
                    st.subheader("Keywords")
                    st.write(", ".join(pattern.get("keywords", [])))
                    
                    st.subheader("Frequency")
                    st.write(f"{pattern.get('frequency', 0)} tickets ({pattern.get('percentage', 0):.1f}% of total)")
                    
                    st.subheader("Sample Ticket Notes")
                    for note in pattern.get("sample_notes", [])[:3]:
                        st.markdown(f"- *{note[:200]}{'...' if len(note) > 200 else ''}*")
    
    # Tab 3: Automation Suggestions
    with tabs[2]:
        st.header("Automation Suggestions")
        
        suggestions = insight_finder_results.get("automation_suggestions", [])
        if not suggestions:
            st.info("No automation suggestions available.")
        else:
            for i, suggestion in enumerate(suggestions):
                # Only display suggestions with a valid problem root cause
                if "problem_root_cause" in suggestion and suggestion["problem_root_cause"]:
                    with st.expander(f"Suggestion {i+1}: {suggestion.get('keywords', [''])[0] if suggestion.get('keywords') else ''}"):
                        # Problem root cause
                        st.subheader("Problem Root Cause")
                        st.write(suggestion["problem_root_cause"])
                        
                        # Suggested solution
                        if "suggested_solution" in suggestion and suggestion["suggested_solution"]:
                            st.subheader("Suggested Solution")
                            st.write(suggestion["suggested_solution"])
                        
                        # Justification
                        if "justification" in suggestion and suggestion["justification"]:
                            st.subheader("Why This is the Best Solution")
                            st.write(suggestion["justification"])
                        
                        # Impact
                        st.subheader("Impact")
                        st.write(f"This solution addresses {suggestion.get('frequency', 0)} tickets ({suggestion.get('percentage', 0):.1f}% of total tickets)")
    
    # Tab 4: Custom Query
    with tabs[3]:
        st.header("Custom Query")
        
        # Add custom query input at the top
        if st.session_state.user_query_agent:
            st.subheader("Ask Your Own Question")
            st.markdown("Enter any question about the ticket data and press tab to submit:")
            custom_query = st.text_area("", height=150, placeholder="e.g., What types of issues take the longest to resolve?", label_visibility="collapsed", key="custom_query_tab", on_change=None)
            
            # Focus on the button when tab is selected
            submit_button = st.button("Submit Question", key="custom_query_submit_btn", use_container_width=True)
            if custom_query and submit_button:
                with st.spinner("Analyzing..."):
                    response = st.session_state.user_query_agent.process_query(custom_query)
                    st.session_state.user_query_history.append(response)
                    st.rerun()
        
        st.markdown("---")
        st.subheader("Query History")
        
        # Check if there are any queries in the history
        if not st.session_state.user_query_history:
            st.info("No queries have been made yet. Use the questions section to analyze the ticket data.")
        else:
            # Display all previous queries with their responses
            for i, query_result in enumerate(reversed(st.session_state.user_query_history)):
                query_text = query_result.get('question', 'Unknown query')
                
                # Create an expander for each query
                with st.expander(f"Q: {query_text}", expanded=(i == 0)):
                    # Display the response
                    st.markdown("### Response")
                    
                    # Clean up response text
                    response_text = query_result.get("response", "No response available.")
                    
                    # Remove thinking process if present
                    if "<think>" in response_text:
                        response_text = response_text.split("<think>")[0].strip()
                        
                    # Remove Response: header if present
                    if response_text.startswith("Response:"):
                        response_text = response_text[9:].strip()
                    
                    # Display the cleaned response
                    st.write(response_text)
                    
                    # Show additional metadata for standard questions
                    if query_result.get("is_standard"):
                        st.caption(f"Standard question ID: {query_result.get('question_id', 'unknown')}")

def main():
    """Main function for the Streamlit app."""
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Main app layout and logic
    try:
        if not st.session_state.file_uploaded:
            # Show welcome message when no file is uploaded
            st.title("Quantitative AI-Powered Ticket Advisor")
            st.markdown("""
            This tool analyzes help desk ticket data to identify automation opportunities.
            
            **To get started:**
            1. Upload your ticket data file (.csv or .xlsx) using the sidebar
            2. Click 'Process Data' to analyze the tickets
            3. Explore the data visualization and automation suggestions
            4. Ask questions to gain deeper insights
            
            **Your ticket data should include:**
            - Ticket ID/Number
            - Description
            - Open/Close dates
            - Resolution notes
            - Assignment information
            """)
            
            # Sidebar for file upload
            with st.sidebar:
                st.subheader("Upload Ticket Data")
                
                # File uploader
                uploaded_file = st.file_uploader("Upload .csv or .xlsx file", type=["csv", "xlsx", "xls"])
                
                if uploaded_file is not None:
                    if st.button("Process Data"):
                        with st.spinner("Processing data..."):
                            success = process_uploaded_file(uploaded_file)
                            if success:
                                st.success("File processed successfully!")
                                st.rerun()
        else:
            # Show results when file is processed
            display_results()
            
            # Sidebar content when data is loaded
            with st.sidebar:

                # Option to upload a new file
                st.subheader("Upload New Data")
                if st.button("Clear Current Data"):
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    initialize_session_state()
                    st.rerun()
                
                uploaded_file = st.file_uploader("Upload .csv or .xlsx file", type=["csv", "xlsx", "xls"])
                if uploaded_file is not None:
                    if st.button("Process Data"):
                        with st.spinner("Processing data..."):
                            success = process_uploaded_file(uploaded_file)
                            if success:
                                st.success("File processed successfully!")
                                st.rerun()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main function: {str(e)}", exc_info=True)

# Run the main function when script is executed
if __name__ == "__main__":
    main()