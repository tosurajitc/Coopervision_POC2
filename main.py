import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import time
import random
import re

# Import your custom agents
from agents.data_processing_agent import DataProcessingAgent
from agents.insight_generation_agent import InsightGenerationAgent
from agents.implementation_strategy_agent import ImplementationStrategyAgent
from agents.user_query_agent import UserQueryAgent

# Enable caching for expensive function calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_data(_data_agent, data, keywords=None):
    """
    Process data with caching to avoid redundant processing
    The leading underscore in _data_agent tells Streamlit not to hash this argument
    """
    # Based on the original code, process_data only takes data as argument
    return _data_agent.process_data(data)

@st.cache_data(ttl=3600)
def answer_question(_query_agent, question, data):
    """
    Cache question answers to avoid redundant API calls
    The leading underscore in _query_agent tells Streamlit not to hash this argument
    """
    return _query_agent.answer_question(question, data)

@st.cache_data(ttl=3600)
def generate_implementation_plan(_implementation_agent, question, answer):
    """
    Cache implementation plans to avoid redundant API calls
    Uses the ImplementationStrategyAgent to generate proper implementation plans
    
    Args:
        _implementation_agent: The implementation strategy agent
        question: The original question or opportunity
        answer: The answer or insight provided by the query agent
        
    Returns:
        A structured implementation plan
    """
    # Use the implementation agent's generate_plan method
    return _implementation_agent.generate_plan(question, answer)

@st.cache_data(ttl=3600)
def generate_insights(_insight_agent, data, num_insights=5):
    """
    Cache insights to avoid redundant API calls
    The leading underscore in _insight_agent tells Streamlit not to hash this argument
    
    Args:
        _insight_agent: The agent for generating insights
        data: The processed data
        num_insights: Number of insights to generate (default: 5)
        
    Returns:
        A list of insight dictionaries
    """
    try:
        # Add debug print statements to help troubleshoot
        print(f"Generating {num_insights} insights from data with {len(data)} rows and columns: {data.columns.tolist()}")
        
        # Explicitly request the number of insights
        insights = _insight_agent.generate_insights(data, num_insights=num_insights)
        
        # More debug information
        print(f"Generated {len(insights)} insights from the agent")
        
        return insights
    except Exception as e:
        print(f"Error in generate_insights caching function: {str(e)}")
        # Return a basic error insight
        return [{
            'title': 'Error Generating Insights',
            'issue': f'An error occurred during insight generation: {str(e)}',
            'solution': 'Please check the logs for more details and try again.',
            'justification': 'The system encountered a technical issue while analyzing your data.',
            'impact': 'low',
            'description': f'Error during insight generation: {str(e)}'
        }]

@st.cache_data(ttl=3600)
def generate_plan_for_opportunity(_implementation_agent, opportunity):
    """
    Cache implementation plans for opportunities
    The leading underscore in _implementation_agent tells Streamlit not to hash this argument
    """
    return _implementation_agent.generate_plan_for_opportunity(opportunity)

def load_environment():
    """
    Load environment variables and set up initial configuration
    """
    load_dotenv()
    
    # Set page configuration
    st.set_page_config(
        page_title="AI-Driven Ticket Analysis Orchestrator",
        page_icon="🤖",
        layout="wide"
    )

def initialize_agents():
    """
    Initialize all required agents with error handling
    """
    try:
        # Add session state to track agent initialization
        if 'agents_initialized' not in st.session_state:
            data_agent = DataProcessingAgent()
            insight_agent = InsightGenerationAgent()
            implementation_agent = ImplementationStrategyAgent()
            query_agent = UserQueryAgent()
            
            # Store agents in session state
            st.session_state['data_agent'] = data_agent
            st.session_state['insight_agent'] = insight_agent
            st.session_state['implementation_agent'] = implementation_agent
            st.session_state['query_agent'] = query_agent
            st.session_state['agents_initialized'] = True
            
        # Retrieve agents from session state
        return (
            st.session_state['data_agent'],
            st.session_state['insight_agent'],
            st.session_state['implementation_agent'],
            st.session_state['query_agent']
        )
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        return None, None, None, None

def handle_file_upload(data_agent):
    """
    Handle ticket data and optional keyword file upload
    
    Args:
        data_agent (DataProcessingAgent): Agent for processing data
    
    Returns:
    - processed_data (DataFrame): Processed ticket data
    - keywords (list): Optional keywords
    """
    # Sidebar for file uploads
    with st.sidebar:
        st.subheader("Upload Data")
        
        # Ticket data upload
        ticket_file = st.file_uploader(
            "Upload ticket data", 
            type=['csv', 'xlsx'], 
            help="Upload your ticket data file"
        )
        
        # Optional keywords file
        keyword_file = st.file_uploader(
            "Upload keywords (optional)", 
            type=['txt'], 
            help="Upload a text file with keywords to focus analysis"
        )
    
    # Process uploaded files
    if ticket_file is not None:
        try:
            # Store file data in session state to avoid re-uploads
            if 'ticket_data' not in st.session_state or st.session_state['ticket_file_name'] != ticket_file.name:
                # Read ticket data
                if ticket_file.name.endswith('.csv'):
                    data = pd.read_csv(ticket_file)
                else:
                    data = pd.read_excel(ticket_file)
                
                st.session_state['ticket_data'] = data
                st.session_state['ticket_file_name'] = ticket_file.name
            else:
                data = st.session_state['ticket_data']
            
            # Process keywords if uploaded
            keywords = None
            if keyword_file is not None:
                if 'keywords' not in st.session_state or st.session_state['keyword_file_name'] != keyword_file.name:
                    keywords = keyword_file.getvalue().decode('utf-8').splitlines()
                    st.session_state['keywords'] = keywords
                    st.session_state['keyword_file_name'] = keyword_file.name
                else:
                    keywords = st.session_state['keywords']
            
            # Process data using data processing agent with caching
            # Note: keywords are collected but not used in processing currently
            with st.spinner("Processing data... This might take a moment."):
                processed_data = process_data(data_agent, data, keywords)
            
            return processed_data, keywords
        
        except Exception as e:
            st.error(f"Error processing files: {e}")
            return None, None
    
    return None, None

def predefined_questions_tab(processed_data, query_agent, implementation_agent):
    """
    Tab for predefined questions with interactive analysis
    
    Args:
        processed_data (pd.DataFrame): Processed ticket data
        query_agent (UserQueryAgent): Agent for handling queries
        implementation_agent (ImplementationStrategyAgent): Agent for generating implementation plans
    """
    st.header("Predefined Automation Analysis Questions")
    
    # Predefined questions
    questions = [
        "What are the most frequently reported pain points in ticket descriptions?",
        "Which ticket resolutions involve repetitive manual steps?",
        "Are there tickets resolvable through self-service options?",
        "Which issues experience delays due to team dependencies?",
        "Are tickets often misrouted or reassigned?",
        "Do users report similar issues that could be proactively prevented?",
        "Which tickets involve extensive manual data entry?",
        "Are there communication gaps between users and support teams?",
        "Do resolution notes indicate recurring workarounds?",
        "Are certain tickets caused by lack of training or unclear processes?"
    ]
    
    # Initialize tracking of selected question
    if 'tab1_current_question_idx' not in st.session_state:
        st.session_state['tab1_current_question_idx'] = 0
        
    # Add selectbox for questions and track changes
    selected_question_idx = st.selectbox(
        "Select a question to analyze:", 
        range(len(questions)),
        format_func=lambda i: f"Question {i+1}: {questions[i]}",
        key="question_selector"
    )
    
    # Check if question has changed
    if selected_question_idx != st.session_state['tab1_current_question_idx']:
        # Question changed, reset analysis state
        st.session_state['tab1_analysis_shown'] = False
        st.session_state['tab1_impl_plan_shown'] = False
        st.session_state['tab1_current_question_idx'] = selected_question_idx
    
    # Get the selected question
    selected_question = questions[selected_question_idx]
    
    # Display the selected question in large, bold text
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <p style="font-size: 300%; font-weight: bold; color: #1E1E1E;">
            {selected_question}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis button and processing
    if 'tab1_analysis_shown' not in st.session_state:
        st.session_state['tab1_analysis_shown'] = False
    
    if 'tab1_processing' not in st.session_state:
        st.session_state['tab1_processing'] = False
        
    # Create a placeholder for the status message
    status_placeholder = st.empty()
    
    # Process analysis in the background when flagged
    if st.session_state.get('tab1_processing', False):
        try:
            # Show a custom spinner with just the text we want
            status_placeholder.markdown("⏳ Analyzing the question... Please wait.")
            
            # Use cached function to get analysis - this happens silently
            analysis = answer_question(query_agent, selected_question, processed_data)
            
            # Store the result and update state
            st.session_state['tab1_analysis'] = analysis
            st.session_state['tab1_analysis_shown'] = True
            st.session_state['tab1_current_question'] = selected_question
        except Exception as e:
            st.error(f"Error processing question: {e}")
            st.session_state['tab1_analysis'] = None
            # Add rate limit handling
            if "429" in str(e):
                st.warning("API rate limit reached. Please wait a moment before trying again.")
                # Add retry logic with exponential backoff
                st.info("Retrying in 10 seconds...")
                time.sleep(10)
        finally:
            # Clear the processing flag and the status message
            st.session_state['tab1_processing'] = False
            status_placeholder.empty()
            # Force a rerun to update the UI
            st.experimental_rerun()
            
    # Process implementation plan in the background when flagged
    if 'tab1_impl_processing' not in st.session_state:
        st.session_state['tab1_impl_processing'] = False
        
    if st.session_state.get('tab1_impl_processing', False):
        try:
            # Show a custom spinner with just the text we want
            status_placeholder.markdown("⏳ Generating implementation plan... Please wait.")
            
            # Get the current question and analysis from session state
            current_question = st.session_state.get('tab1_current_question', selected_question)
            current_analysis = st.session_state.get('tab1_analysis', '')
            
            # Generate implementation plan using the implementation agent - this happens silently
            implementation_plan = generate_implementation_plan(
                implementation_agent,  # Use implementation agent instead of query agent
                current_question, 
                current_analysis  # Pass the analysis as the second parameter
            )
            
            # Store the result and update state
            st.session_state['tab1_impl_plan'] = implementation_plan
            st.session_state['tab1_impl_plan_shown'] = True
        except Exception as e:
            st.error(f"Error generating implementation plan: {e}")
            st.session_state['tab1_impl_plan'] = None
        finally:
            # Clear the processing flag and the status message
            st.session_state['tab1_impl_processing'] = False
            status_placeholder.empty()
            # Force a rerun to update the UI
            st.experimental_rerun()
    
    # Show analysis button if analysis is not shown yet
    col1, col2 = st.columns([1, 3])
    with col1:
        if not st.session_state['tab1_analysis_shown']:
            if st.button("Get Analysis", key="btn_get_analysis"):
                # Set the processing flag to trigger the background operation
                st.session_state['tab1_processing'] = True
                st.experimental_rerun()
        else:
            # Show reset button when analysis is displayed
            if st.button("New Analysis", key="btn_new_analysis"):
                st.session_state['tab1_analysis_shown'] = False
                st.session_state['tab1_impl_plan_shown'] = False
                st.experimental_rerun()
    
    # Display analysis if it has been generated
    if st.session_state.get('tab1_analysis_shown', False) and not st.session_state.get('tab1_processing', False):
        analysis = st.session_state.get('tab1_analysis')
        
        if analysis:
            st.markdown("### Analysis")
            st.markdown(analysis)
            
            # Implementation plan button
            plan_key = "tab1_impl_plan_shown"
            if plan_key not in st.session_state:
                st.session_state[plan_key] = False
                
            if not st.session_state[plan_key] and not st.session_state.get('tab1_impl_processing', False):
                if st.button("Get Implementation Plan", key="btn_tab1_impl_plan"):
                    # Set the processing flag to trigger the background operation
                    st.session_state['tab1_impl_processing'] = True
                    st.experimental_rerun()
            
            # Display implementation plan if it has been generated
            if st.session_state.get(plan_key, False):
                impl_plan = st.session_state.get('tab1_impl_plan')
                if impl_plan:
                    st.markdown("### Implementation Plan")
                    st.markdown(impl_plan)

def automation_opportunities_tab(processed_data, insight_agent, implementation_agent):
    """
    Tab for displaying automation opportunities with a structured format
    
    Args:
        processed_data (pd.DataFrame): Processed ticket data
        insight_agent (InsightGenerationAgent): Agent for generating insights
        implementation_agent (ImplementationStrategyAgent): Agent for generating implementation plans
    """
    st.header("Automated Insights and Opportunities")
    
    # Reset tab1 state when visiting tab2
    if 'tab1_analysis_shown' in st.session_state:
        st.session_state['tab1_analysis_shown'] = False
    if 'tab1_impl_plan_shown' in st.session_state:
        st.session_state['tab1_impl_plan_shown'] = False
    
    # Reset tab3 state when visiting tab3
    if 'tab3_analysis_shown' in st.session_state:
        st.session_state['tab3_analysis_shown'] = False
    if 'tab3_impl_plan_shown' in st.session_state:
        st.session_state['tab3_impl_plan_shown'] = False
    
    # Status placeholder for feedback during processing
    status_placeholder = st.empty()
    
    # Initialize state variables for insights generation
    if 'tab2_insights_shown' not in st.session_state:
        st.session_state['tab2_insights_shown'] = False
    
    if 'tab2_processing' not in st.session_state:
        st.session_state['tab2_processing'] = False
    
    # Generate insights when the processing flag is set
    if st.session_state.get('tab2_processing', False):
        try:
            # Show processing message
            status_placeholder.markdown("⏳ Generating insights from ticket data... Please wait. This may take a few moments.")
            
            # Call the insight generation agent with proper parameters
            opportunities = generate_insights(insight_agent, processed_data, num_insights=5)
            
            # Store the results in session state
            st.session_state['tab2_opportunities'] = opportunities
            st.session_state['tab2_insights_shown'] = True
            
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            st.session_state['tab2_opportunities'] = None
            # Add rate limit handling
            if "429" in str(e):
                st.warning("API rate limit reached. Please wait a moment before trying again.")
                # Add retry logic with exponential backoff
                st.info("Retrying in 10 seconds...")
                time.sleep(10)
                st.experimental_rerun()
        finally:
            # Clear the processing flag and the status message
            st.session_state['tab2_processing'] = False
            status_placeholder.empty()
            # Force a rerun to update the UI
            st.experimental_rerun()
    
    # Button to generate insights
    if not st.session_state['tab2_insights_shown']:
        if st.button("Generate Insights", key="btn_generate_insights"):
            # Set the processing flag to trigger the background operation
            st.session_state['tab2_processing'] = True
            st.experimental_rerun()
    
    # Display opportunities if they have been generated
    if st.session_state.get('tab2_insights_shown', False) and not st.session_state.get('tab2_processing', False):
        opportunities = st.session_state.get('tab2_opportunities')
        
        if opportunities:
            # Debug information to make sure we have valid opportunities
            st.success(f"Found {len(opportunities)} automation opportunities in your ticket data")
            
            # Display all opportunities as structured expandable blocks
            for i, opportunity in enumerate(opportunities):
                # Extract title for the expander - use the opportunity title
                opportunity_title = opportunity.get('title', f"Opportunity {i+1}")
                
                # If the title is very long, truncate it for the expander
                display_title = opportunity_title
                if len(display_title) > 80:  # Truncate if too long
                    display_title = display_title[:77] + "..."
                
                # Use the title in the expander heading
                with st.expander(f"#{i+1}: {display_title}", expanded=True):
                    # Full title (even if truncated in expander)
                    st.markdown(f"### {opportunity_title}")
                    
                    # Display structured information
                    st.markdown("#### Issue Identified")
                    if 'issue' in opportunity:
                        st.markdown(opportunity['issue'])
                    else:
                        # Extract issue from description if not available as separate field
                        st.markdown(opportunity.get('description', '').split('\n')[0] if '\n' in opportunity.get('description', '') else opportunity.get('description', ''))
                    
                    st.markdown("#### Proposed Solution")
                    if 'solution' in opportunity:
                        st.markdown(opportunity['solution'])
                    else:
                        # Extract solution if not available
                        parts = opportunity.get('description', '').split('\n')
                        st.markdown(parts[1] if len(parts) > 1 else "Solution details included in implementation plan")
                    
                    st.markdown("#### Justification")
                    if 'justification' in opportunity:
                        st.markdown(opportunity['justification'])
                    else:
                        # Extract justification if not available
                        parts = opportunity.get('description', '').split('\n')
                        st.markdown(parts[2] if len(parts) > 2 else "See implementation plan for full justification")
                    
                    st.markdown("#### Automation Impact")
                    if 'impact' in opportunity:
                        st.markdown(opportunity['impact'])
                    else:
                        # Extract impact if not available
                        parts = opportunity.get('description', '').split('\n')
                        st.markdown(parts[3] if len(parts) > 3 else "Impact assessment included in implementation plan")
                    
                    # Debugging information without using nested expanders
                    # st.markdown("#### Debug Information")
                    # st.write(f"Opportunity Type: {type(opportunity)}")
                    # st.write(f"Fields: {', '.join(opportunity.keys())}")
                    
                    # Implementation plan with unique key
                    plan_key = f"tab2_impl_plan_shown_{i}"
                    impl_processing_key = f"tab2_impl_processing_{i}"
                    
                    # Initialize plan processing state
                    if impl_processing_key not in st.session_state:
                        st.session_state[impl_processing_key] = False
                    
                    if plan_key not in st.session_state:
                        st.session_state[plan_key] = False
                    
                    # Process implementation plan in the background when flagged
                    if st.session_state.get(impl_processing_key, False):
                        try:
                            # Show a custom spinner with just the text we want
                            impl_status = st.empty()
                            impl_status.markdown("⏳ Generating implementation plan... Please wait.")
                            
                            # Generate plan for this opportunity
                            plan = generate_plan_for_opportunity(implementation_agent, opportunity)
                            st.session_state[f'tab2_impl_plan_{i}'] = plan
                            st.session_state[plan_key] = True
                        except Exception as e:
                            st.error(f"Error generating implementation plan: {e}")
                            st.session_state[f'tab2_impl_plan_{i}'] = None
                        finally:
                            # Clear the processing flag and the status message
                            st.session_state[impl_processing_key] = False
                            impl_status.empty()
                            # Force a rerun to update the UI
                            st.experimental_rerun()
                    
                    # Implementation plan button if not already generated
                    if not st.session_state[plan_key] and not st.session_state.get(impl_processing_key, False):
                        if st.button(f"Get Implementation Plan", key=f"btn_tab2_impl_plan_{i}"):
                            # Set the processing flag to trigger the background operation
                            st.session_state[impl_processing_key] = True
                            st.experimental_rerun()
                    
                    # Display implementation plan if it has been generated
                    if st.session_state.get(plan_key, False):
                        impl_plan = st.session_state.get(f'tab2_impl_plan_{i}')
                        if impl_plan:
                            st.markdown("#### Implementation Plan")
                            st.markdown(impl_plan)
        else:
            st.warning("No automation opportunities were found. This might indicate an issue with the insight generation process or that the data doesn't contain clear automation patterns.")
        
        # Add reset button
        if st.button("Reset Insights", key="btn_reset_tab2"):
            # Clear all tab2 state variables
            tab2_keys = [key for key in st.session_state.keys() if key.startswith('tab2_')]
            for key in tab2_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['tab2_insights_shown'] = False
            st.experimental_rerun()
    elif not st.session_state.get('tab2_insights_shown', False) and not st.session_state.get('tab2_processing', False):
        # Show instructions when no insights have been generated yet
        st.info("Click the 'Generate Insights' button to analyze your ticket data and identify automation opportunities.")

def custom_query_tab(processed_data, query_agent, implementation_agent):
    """
    Tab for custom user queries
    
    Args:
        processed_data (pd.DataFrame): Processed ticket data
        query_agent (UserQueryAgent): Agent for handling queries
        implementation_agent (ImplementationStrategyAgent): Agent for generating implementation plans
    """
    st.header("Custom Query Analysis")
    
    # Reset tab1 state when visiting tab3
    if 'tab1_analysis_shown' in st.session_state:
        st.session_state['tab1_analysis_shown'] = False
    if 'tab1_impl_plan_shown' in st.session_state:
        st.session_state['tab1_impl_plan_shown'] = False
    
    # Reset tab2 state when visiting tab3
    if 'tab2_insights_shown' in st.session_state:
        st.session_state['tab2_insights_shown'] = False
    if 'tab2_impl_plan_shown' in st.session_state:
        st.session_state['tab2_impl_plan_shown'] = False
    
    # Custom query input
    query = st.text_area("Enter your specific query about the ticket data", key="custom_query_input")
    
    # Analysis button with unique key
    if 'tab3_analysis_shown' not in st.session_state:
        st.session_state['tab3_analysis_shown'] = False
        
    if not st.session_state['tab3_analysis_shown']:
        if st.button("Analyze Query", key="btn_analyze_query"):
            if query:
                st.session_state['tab3_analysis_shown'] = True
                st.session_state['tab3_current_query'] = query
                
                with st.spinner("Analyzing your query... This might take a moment."):
                    try:
                        # Generate response with caching
                        response = answer_question(query_agent, query, processed_data)
                        st.session_state['tab3_analysis'] = response
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        st.session_state['tab3_analysis'] = None
                        # Add rate limit handling
                        if "429" in str(e):
                            st.warning("API rate limit reached. Please wait a moment before trying again.")
                            # Add retry logic with exponential backoff
                            st.info("Retrying in 10 seconds...")
                            time.sleep(10)
                            st.experimental_rerun()
            else:
                st.warning("Please enter a query")
    
    # Display analysis if it has been generated
    if st.session_state.get('tab3_analysis_shown', False):
        current_query = st.session_state.get('tab3_current_query')
        analysis = st.session_state.get('tab3_analysis')
        
        if analysis:
            st.markdown("### Query Analysis")
            st.markdown(analysis)
            
            # Implementation plan with unique key
            plan_key = "tab3_impl_plan_shown"
            if plan_key not in st.session_state:
                st.session_state[plan_key] = False
                
            if not st.session_state[plan_key]:
                if st.button("Get Implementation Plan", key="btn_tab3_impl_plan"):
                    st.session_state[plan_key] = True
                    
                    with st.spinner("Generating implementation plan..."):
                        try:
                            implementation_plan = generate_implementation_plan(
                                implementation_agent,  # Use implementation_agent here
                                current_query, 
                                analysis  # Pass the analysis instead of processed_data
                            )
                            st.session_state['tab3_impl_plan'] = implementation_plan
                        except Exception as e:
                            st.error(f"Error generating implementation plan: {e}")
                            st.session_state['tab3_impl_plan'] = None
            
            # Display implementation plan if it has been generated
            if st.session_state.get(plan_key, False):
                impl_plan = st.session_state.get('tab3_impl_plan')
                if impl_plan:
                    st.markdown("### Implementation Plan")
                    st.markdown(impl_plan)
        
        # Add reset button
        if st.button("Reset Analysis", key="btn_reset_tab3"):
            st.session_state['tab3_analysis_shown'] = False
            if 'tab3_impl_plan_shown' in st.session_state:
                st.session_state['tab3_impl_plan_shown'] = False
            st.experimental_rerun()


def display_data_requirements_guide():
    """
    Displays guidance about data requirements and quality for optimal analysis
    """
    with st.sidebar.expander("📊 Data Requirements Guide", expanded=False):
        st.markdown("""
        ### Required Fields
        For optimal analysis, your ticket data should include these fields:
        
        - **Ticket ID**: Unique identifier for each ticket
        - **Created Date**: When the ticket was opened
        - **Status**: Current status (Open, Closed, In Progress, etc.)
        - **Category/Type**: The type or category of the ticket
        - **Description**: Detailed description of the issue
        - **Resolution Notes**: How the ticket was resolved
        - **Resolution Time**: Time taken to resolve the ticket
        - **Assigned Team/Agent**: Who handled the ticket
        - **Priority**: Ticket priority level
        
        ### Data Quality Tips
        
        Higher quality data leads to better automation insights:
        
        1. **Complete Descriptions**: Tickets with detailed descriptions yield more accurate analysis
        2. **Resolution Details**: Clear notes about resolution steps help identify automation opportunities
        3. **Consistent Categories**: Well-categorized tickets improve pattern recognition
        4. **Volume Matters**: More tickets provide better statistical insights
        5. **Time Spans**: Data covering longer periods helps identify recurring issues
        
        ### How Data Quality Affects Results
        
        The quality of your ticket data directly impacts:
        
        - **Automation Potential**: Better data reveals more opportunities
        - **Implementation Precision**: Clearer patterns lead to more specific implementation plans
        - **ROI Calculation**: More detailed data enables better impact estimates
        - **Issue Identification**: Comprehensive data helps find hidden problems
        """)

def add_data_quality_section(processed_data):
    """
    Adds a section explaining how data quality affects automation insights
    
    Args:
        processed_data (pd.DataFrame): The processed ticket data
    """
    with st.expander("💡 How Data Quality Affects Automation Insights", expanded=False):
        st.markdown("""
        ### Improving Insights Through Better Data
        
        The quality and completeness of your ticket data directly impacts the value of the automation insights generated. Here's how better data leads to better automation opportunities:
        
        #### 1. Pattern Recognition
        High-quality data with consistent categorization and detailed descriptions allows the system to identify recurring patterns that might be missed in sparse or inconsistent data.
        
        #### 2. Root Cause Analysis
        When tickets contain comprehensive problem descriptions and resolution notes, the system can better identify the underlying causes rather than just symptoms, leading to more impactful automation solutions.
        
        #### 3. Process Bottlenecks
        Complete timestamp data and assignment information help identify where delays occur in your support process, highlighting opportunities for workflow automation.
        
        #### 4. Resolution Pathway Optimization
        Detailed resolution steps in your data allow the system to identify common solutions that could be standardized or automated.
        
        #### 5. Self-Service Opportunities
        Tickets with clear user context help identify issues that could be resolved through self-service options, reducing ticket volume altogether.
        
        #### Data Enrichment Recommendations
        
        If your current analysis seems limited, consider enriching your ticket data with:
        
        - More detailed resolution steps
        - Categorization of manual vs. automated steps taken
        - Time tracking for each resolution stage
        - User feedback on solutions provided
        - Cross-references to related tickets or recurring issues
        """)
        
        # Add data statistics if available
        if processed_data is not None:
            try:
                num_tickets = len(processed_data)
                num_columns = len(processed_data.columns)
                date_range = "Unknown"
                
                # Try to determine date range if date column exists
                date_columns = [col for col in processed_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    # Convert first date column to datetime and get range
                    try:
                        dates = pd.to_datetime(processed_data[date_columns[0]])
                        min_date = dates.min().strftime('%Y-%m-%d')
                        max_date = dates.max().strftime('%Y-%m-%d')
                        date_range = f"{min_date} to {max_date}"
                    except:
                        pass
                
                st.markdown("### Your Current Data Statistics")
                st.markdown(f"""
                - **Number of Tickets**: {num_tickets}
                - **Number of Data Fields**: {num_columns}
                - **Date Range**: {date_range}
                """)
                
                # Completeness assessment
                st.markdown("### Data Completeness")
                completeness = processed_data.notna().mean() * 100
                completeness_df = pd.DataFrame({
                    'Field': completeness.index,
                    'Completeness %': completeness.values
                })
                completeness_df = completeness_df.sort_values('Completeness %', ascending=False)
                st.dataframe(completeness_df.style.format({'Completeness %': '{:.1f}%'}))
                
                # Recommendation based on completeness
                low_completeness = completeness_df[completeness_df['Completeness %'] < 90]
                if not low_completeness.empty:
                    st.markdown("### Recommendation")
                    st.markdown(f"""
                    Consider improving data completeness for these fields to enhance analysis quality:
                    - {", ".join(low_completeness['Field'].head(3).tolist())}
                    """)
            except:
                st.markdown("Unable to analyze current data statistics.")            
def display_sample_data_info():
    """
    Function to display information about the sample ticket data
    """
    with st.expander("ℹ️ About the Sample Ticket Data", expanded=False):
        st.markdown("""
        ## Sample Ticket Data Information
        
        The provided sample ticket data represents high-quality support ticket information that will produce excellent automation insights. Here's what makes this sample data particularly valuable:
        
        ### Key Data Quality Attributes
        
        1. **Comprehensive Fields**: The dataset includes all critical fields for analysis:
           - Basic tracking (TicketID, dates, status)
           - Classification (Category, Priority)
           - Assignment (Team, Agent)
           - Customer information
           - Detailed description and resolution notes
           - Performance metrics (resolution time, reopen count, satisfaction)
        
        2. **Detailed Descriptions**: Each ticket includes specific issue descriptions instead of vague summaries.
           
        3. **Thorough Resolution Notes**: The notes explain exactly what was done to resolve each issue, including:
           - Root cause identification
           - Specific actions taken
           - Verification steps
           - Additional measures (training, documentation, monitoring)
        
        4. **Consistent Categorization**: Tickets are properly categorized into meaningful groups like:
           - System Access
           - Software Issue
           - Hardware Issue
           - Network Issue
           - Data Request
        
        5. **Complete Metrics**: Important time and quality metrics are included:
           - Resolution time in hours
           - Whether tickets were reopened
           - Customer satisfaction scores
        
        ### How This Data Enables Better Insights
        
        With this level of data quality, the AI can identify:
        
        1. **Patterns in Resolution Steps**: The detailed resolution notes reveal repeated processes that could be automated.
        
        2. **Common Root Causes**: Many tickets show underlying causes rather than just symptoms.
        
        3. **Self-Service Opportunities**: Tickets that involve simple instructions or guidance are easily identifiable.
        
        4. **Process Inefficiencies**: Resolution times combined with detailed notes highlight bottlenecks.
        
        5. **Cross-Team Dependencies**: Assignment information shows when issues require coordination between teams.
        
        ### How to Use This Sample Data
        
        1. Copy this data to an Excel file and save it as CSV or XLSX
        2. Upload the file using the uploader in the sidebar
        3. Explore the automation insights generated from this high-quality data
        4. Use this as a template for structuring your own ticket data
        
        The insights generated from this sample data will demonstrate the full capabilities of the analysis system when provided with comprehensive, well-structured information.
        """)

# Function that could be called to show how to download the sample data
def add_sample_data_download_option():
    """
    Adds an option to download sample ticket data
    """
    st.sidebar.markdown("### Sample Data")
    
    # Create a text string of the CSV data
    sample_data = """TicketID,CreatedDate,ClosedDate,Status,Category,Priority,AssignedTeam,AssignedAgent,CustomerID,Description,ResolutionNotes,ResolutionTime,ReopenCount,CustomerSatisfaction
TKT-1001,2023-01-05 09:15:22,2023-01-05 14:37:46,Closed,System Access,High,IT Support,John Smith,CUST-342,User unable to login to the accounting system after password reset. Multiple attempts result in account lockout.,Reset user account in Active Directory and provided temporary password. Guided user through password change process. Confirmed successful login.,5.37,0,4
# ... and so on with the remaining 49 records
"""
    
    st.sidebar.download_button(
        label="Download Sample Data",
        data=sample_data,
        file_name="sample_ticket_data.csv",
        mime="text/csv",
    )
    
    st.sidebar.markdown("Download and use this sample data to see optimal results from the analysis.")


def main():
    """
    Main orchestration function
    """
    # Initialize environment and agents
    load_environment()
    
    # Add a progress bar for initial loading
    with st.spinner("Initializing application..."):
        data_agent, insight_agent, implementation_agent, query_agent = initialize_agents()
    
    # Validate agent initialization
    if any(agent is None for agent in [data_agent, insight_agent, implementation_agent, query_agent]):
        st.error("Failed to initialize one or more agents. Please check your configuration.")
        return
    
    # Title
    st.title("AI-Driven Ticket Analysis Orchestrator")
    
    # Display data requirements guide in sidebar
    display_data_requirements_guide()
    
    # File upload
    processed_data, keywords = handle_file_upload(data_agent)
    
    # Tabs for different functionalities
    if processed_data is not None:
        # Add data quality section at the top
        add_data_quality_section(processed_data)
        
        # Initialize active tab index if not present
        if 'tab_index' not in st.session_state:
            st.session_state['tab_index'] = 0

        # Use integer indices for the radio button
        tab_options = ["Predefined Questions", "Automation Opportunities", "Custom Query"]
        selected_tab = st.radio(
            "Select Tab",
            options=range(len(tab_options)),
            format_func=lambda i: tab_options[i],
            horizontal=True,
            index=st.session_state['tab_index'],
            key="tab_selector"
        )
        
        # Update the tab index in session state
        st.session_state['tab_index'] = selected_tab
        
        # Check if tab has changed and reset state if needed
        previous_tab = st.session_state.get('previous_tab_index', 0)
        if previous_tab != selected_tab:
            # Reset states for all tabs when switching
            tab_state_keys = [
                key for key in st.session_state.keys() 
                if key.startswith('tab1_') or key.startswith('tab2_') or key.startswith('tab3_')
            ]
            for key in tab_state_keys:
                if key.endswith('_shown'):
                    st.session_state[key] = False
            
            # Store the current tab as previous for next check
            st.session_state['previous_tab_index'] = selected_tab
        
        # Show the selected tab content - now passing implementation_agent to all tab functions
        if selected_tab == 0:
            predefined_questions_tab(processed_data, query_agent, implementation_agent)
        elif selected_tab == 1:
            automation_opportunities_tab(processed_data, insight_agent, implementation_agent)
        else:
            custom_query_tab(processed_data, query_agent, implementation_agent)
    else:
        # Display welcome message and instructions when no data is uploaded
        st.info("Please upload ticket data to begin analysis")
        
        st.markdown("""
        ## Welcome to the AI-Driven Ticket Analysis Orchestrator
        
        This tool helps you analyze support ticket data to identify automation opportunities and generate actionable insights.
        
        ### Getting Started
        
        1. **Upload your ticket data** using the file uploader in the sidebar
        2. **Optional**: Upload a keywords file to focus the analysis on specific areas
        3. Once your data is processed, explore the three analysis tabs:
           - **Predefined Questions**: Get answers to common automation questions
           - **Automation Opportunities**: See structured insights and implementation plans
           - **Custom Query**: Ask your own specific questions about the ticket data
        
        ### Data Requirements
        
        For best results, your ticket data should include fields like ticket description, resolution notes, 
        timestamps, categories, and resolution steps. See the **Data Requirements Guide** in the sidebar for more details.
        
        ### How It Works
        
        This tool uses advanced AI to analyze patterns in your support tickets, identify repetitive tasks,
        and suggest automation opportunities with detailed implementation plans.
        """)

if __name__ == "__main__":
    main()