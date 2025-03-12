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
    /* Style for selectbox */
    div[data-baseweb="select"] {
        min-height: 60px;
        max-width: 800px !important;
        width: 100% !important;
    }
    div[data-baseweb="select"] span {
        max-width: 100%;
        white-space: normal;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: visible;
        font-size: 1.6rem !important;
        font-weight: bold !important;
        line-height: 1.5 !important;
    }
    div[data-baseweb="popover"] {
        width: auto !important;
        max-width: 800px !important;
    }
    div[data-baseweb="popover"] div[role="listbox"] {
        max-width: 800px;
        width: auto !important;
        font-size: 1.6rem !important;
    }
    div[data-baseweb="popover"] div[role="option"] {
        white-space: normal;
        padding: 12px;
        line-height: 1.5;
        font-size: 1.6rem !important;
        font-weight: bold !important;
    }
    /* Make selectbox text larger */
    .stSelectbox {
        font-size: 1.6rem !important;
        width: auto !important;
        max-width: 800px !important;
    }
    .stSelectbox > div {
        width: auto !important;
        max-width: 800px !important;
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
    tab_labels = ["Qualitative Analysis", "Patterns", "Automation Suggestions", "💬 Custom Query"]
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
            st.markdown('<h2 style="font-size: 2.4rem; color: #1f4e79; font-weight: bold;">Standard Questions</h2>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.3rem; margin-bottom: 15px;">Select a question to analyze your ticket data and get Qualititave Response:</div>', unsafe_allow_html=True)
            
            # Create a dropdown with questions for selection
            question_options = ["Select a question..."] + [q["question"] for q in standard_questions]
            
            # Style the dropdown container to be larger to accommodate the bigger text
            st.markdown('<div style="margin-bottom: 30px; margin-top: 15px;">', unsafe_allow_html=True)
            
            # Auto-answer on selection change - only create one dropdown
            previous_question = st.session_state.get("previous_question", "")
            selected_question = st.selectbox(
                "", 
                question_options, 
                label_visibility="collapsed", 
                key="data_overview_question_select",
                index=0 if previous_question == "" else question_options.index(previous_question) if previous_question in question_options else 0
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detect when a new question is selected
            if selected_question != "Select a question..." and selected_question != previous_question:
                # Store the current selection for future comparison
                st.session_state.previous_question = selected_question
                
                # Find the question ID
                question_id = None
                for q in standard_questions:
                    if q["question"] == selected_question:
                        question_id = q["id"]
                        st.info(q["description"])
                        break
                
                # Auto-answer the question without a button
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
            st.markdown('<h1 style="font-size: 1.8rem; color: #1f4e79; margin-bottom: 0.8rem;">Latest Response</h1>', unsafe_allow_html=True)
            latest_query = st.session_state.user_query_history[-1]
            
            # Display question in a colored box with larger font to match heading size
            st.markdown(f"""
            <div style="background-color:#e6f0ff; padding:20px; border-radius:5px; margin-bottom:20px; font-size: 1.8rem; font-weight: bold;">
                <strong style="font-size: 2rem;">Question:</strong> {latest_query.get('question', 'Unknown query')}
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
        
        # Create a table for the overview data
        overview_data = []
        
        # File information
        file_info = data_processor_results.get("file_info", {})
        if file_info:
            overview_data.append(["Total Tickets", str(file_info.get("rows", 0))])
            overview_data.append(["File Size", f"{file_info.get('size_mb', 0):.2f} MB"])
        
        # Summary statistics
        summary_stats = data_processor_results.get("summary_stats", {})
        if summary_stats:
            # Resolution time metrics
            if "avg_resolution_time" in summary_stats:
                overview_data.append(["Avg. Resolution Time", f"{summary_stats['avg_resolution_time']:.2f} hrs"])
            if "median_resolution_time" in summary_stats:
                overview_data.append(["Median Resolution Time", f"{summary_stats['median_resolution_time']:.2f} hrs"])
            if "max_resolution_time" in summary_stats:
                overview_data.append(["Max Resolution Time", f"{summary_stats['max_resolution_time']:.2f} hrs"])
        
        # Display data rows without headers
        for row in overview_data:
            st.markdown(f"<div style='display: flex; border-bottom: 1px solid #ddd; padding: 8px 0;'>"
                        f"<div style='flex: 1; font-size: 1.2rem; font-weight: bold;'>{row[0]}</div>"
                        f"<div style='flex: 1; font-size: 1.2rem;'>{row[1]}</div>"
                        f"</div>", 
                       unsafe_allow_html=True)
        
        # Add some space after the table
        st.markdown("<br>", unsafe_allow_html=True)
            
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
    # Tab 4: Custom Query
    with tabs[3]:
        st.header("Custom Query")
        
        # Add custom query input at the top
        if st.session_state.user_query_agent:
            st.subheader("Ask Your Own Question")
            st.markdown("Enter any question about the ticket data and press tab to submit:")
            
            # Create a form to properly handle submission
            with st.form(key="custom_query_form"):
                custom_query = st.text_area("Question", height=150, 
                                          placeholder="e.g., What types of issues take the longest to resolve?", 
                                          label_visibility="collapsed")
                
                # Submit button inside the form
                submit_button = st.form_submit_button("Submit Question", use_container_width=True)
            
            # Process the form submission
            if submit_button and custom_query:
                with st.spinner("Analyzing..."):
                    try:
                        # Process the query and get a response
                        response = st.session_state.user_query_agent.process_query(custom_query)
                        
                        # Add to query history
                        st.session_state.user_query_history.append(response)
                        
                        # Display the response immediately
                        st.markdown("### Response")
                        
                        # Display question in a colored box
                        st.markdown(f"""
                        <div style="background-color:#e6f0ff; padding:15px; border-radius:5px; margin-bottom:15px;">
                            <strong>Question:</strong> {custom_query}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Clean up response text
                        response_text = response.get("response", "No response available.")
                        if "<think>" in response_text:
                            response_text = response_text.split("<think>")[0].strip()
                        if response_text.startswith("Response:"):
                            response_text = response_text[9:].strip()
                        
                        # Display the response
                        st.write(response_text)
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        logger.error(f"Error in custom query: {str(e)}", exc_info=True)

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
            2. Check pre defined 10 Qualitative questions from the drop down
            3. Explore the data pattern and automation suggestions
            4. Ask questions to gain deeper insights
            
            **Your ticket data should include:**
            - Ticket ID/Number
            - Description - as detailed as possible
            - Open/Close dates
            - Resolution notes - as detailed as possible
            - Assignment information
            - Most importantly detailed Closed notes
            """)
            
            # Sidebar for file upload
            with st.sidebar:
                st.subheader("Upload Ticket Data")
                
                # File uploader - auto-process when file is uploaded
                uploaded_file = st.file_uploader("Upload .csv or .xlsx file", type=["csv", "xlsx", "xls"], 
                                             key="welcome_file_uploader", help="Select a ticket data file to analyze")
                
                # Auto-process the file when uploaded
                if uploaded_file is not None:
                    # Check if this is a new file that hasn't been processed yet
                    current_file = getattr(st.session_state, 'current_file', None)
                    if current_file != uploaded_file.name:
                        st.session_state.current_file = uploaded_file.name
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
                
                # File uploader with auto-processing
                uploaded_file = st.file_uploader("Upload .csv or .xlsx file", type=["csv", "xlsx", "xls"], 
                                              key="sidebar_file_uploader")
                
                # Auto-process the file when uploaded
                if uploaded_file is not None:
                    # Check if this is a new file that hasn't been processed yet
                    current_file = getattr(st.session_state, 'current_sidebar_file', None)
                    if current_file != uploaded_file.name:
                        st.session_state.current_sidebar_file = uploaded_file.name
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