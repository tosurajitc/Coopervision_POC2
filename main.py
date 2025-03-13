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
def generate_implementation_plan(_query_agent, question, data):
    """
    Cache implementation plans to avoid redundant API calls
    The leading underscore in _query_agent tells Streamlit not to hash this argument
    """
    # Make sure the implementation plan doesn't include thinking blocks
    return _query_agent.answer_question(f"Provide a concise implementation plan for: {question}", data)

@st.cache_data(ttl=3600)
def generate_insights(_insight_agent, data):
    """
    Cache insights to avoid redundant API calls
    The leading underscore in _insight_agent tells Streamlit not to hash this argument
    """
    return _insight_agent.generate_insights(data)

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

def predefined_questions_tab(processed_data, query_agent):
    """
    Tab for predefined questions with interactive analysis
    
    Args:
        processed_data (pd.DataFrame): Processed ticket data
        query_agent (UserQueryAgent): Agent for handling queries
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
    
    # Add selectbox for questions to reduce load
    selected_question_idx = st.selectbox(
        "Select a question to analyze:", 
        range(len(questions)),
        format_func=lambda i: f"Question {i+1}: {questions[i]}",
        key="question_selector"
    )
    
    # Only load analysis for selected question
    selected_question = questions[selected_question_idx]
    
    # Get analysis with a button to avoid double processing
    if 'tab1_analysis_shown' not in st.session_state:
        st.session_state['tab1_analysis_shown'] = False
        
    if not st.session_state['tab1_analysis_shown']:
        if st.button("Get Analysis", key="btn_get_analysis"):
            st.session_state['tab1_analysis_shown'] = True
            st.session_state['tab1_current_question'] = selected_question
            st.session_state['tab1_current_question_idx'] = selected_question_idx
            
            with st.spinner(f"Analyzing question: {selected_question}"):
                try:
                    # Use cached function to get analysis
                    analysis = answer_question(query_agent, selected_question, processed_data)
                    st.session_state['tab1_analysis'] = analysis
                except Exception as e:
                    st.error(f"Error processing question: {e}")
                    st.session_state['tab1_analysis'] = None
                    # Add rate limit handling
                    if "429" in str(e):
                        st.warning("API rate limit reached. Please wait a moment before trying again.")
                        # Add retry logic with exponential backoff
                        st.info("Retrying in 10 seconds...")
                        time.sleep(10)
                        st.experimental_rerun()
    
    # Display analysis if it has been generated
    if st.session_state.get('tab1_analysis_shown', False):
        current_q_idx = st.session_state.get('tab1_current_question_idx')
        current_q = st.session_state.get('tab1_current_question')
        analysis = st.session_state.get('tab1_analysis')
        
        if analysis:
            st.markdown(f"### Analysis for Question {current_q_idx+1}")
            st.markdown(analysis)
            
            # Implementation plan button with unique key
            plan_key = "tab1_impl_plan_shown"
            if plan_key not in st.session_state:
                st.session_state[plan_key] = False
                
            if not st.session_state[plan_key]:
                if st.button("Get Implementation Plan", key="btn_tab1_impl_plan"):
                    st.session_state[plan_key] = True
                    
                    with st.spinner("Generating implementation plan..."):
                        try:
                            # Use cached function with modified prompt to avoid thinking block
                            implementation_plan = generate_implementation_plan(
                                query_agent,
                                current_q, 
                                processed_data
                            )
                            st.session_state['tab1_impl_plan'] = implementation_plan
                        except Exception as e:
                            st.error(f"Error generating implementation plan: {e}")
                            st.session_state['tab1_impl_plan'] = None
            
            # Display implementation plan if it has been generated
            if st.session_state.get(plan_key, False):
                impl_plan = st.session_state.get('tab1_impl_plan')
                if impl_plan:
                    st.markdown("### Implementation Plan")
                    st.markdown(impl_plan)
        
        # Add reset button
        if st.button("Reset Analysis", key="btn_reset_tab1"):
            st.session_state['tab1_analysis_shown'] = False
            if 'tab1_impl_plan_shown' in st.session_state:
                st.session_state['tab1_impl_plan_shown'] = False
            st.experimental_rerun()

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
    
    # Add progress indicator
    if 'tab2_insights_shown' not in st.session_state:
        st.session_state['tab2_insights_shown'] = False
        
    if not st.session_state['tab2_insights_shown']:
        if st.button("Generate Insights", key="btn_generate_insights"):
            st.session_state['tab2_insights_shown'] = True
            with st.spinner("Generating insights... This might take a moment."):
                try:
                    # Explicitly request 5 insights
                    # First check if the generate_insights method accepts a num_insights parameter
                    import inspect
                    sig = inspect.signature(insight_agent.generate_insights)
                    
                    if 'num_insights' in sig.parameters:
                        # If the method accepts num_insights, pass it
                        opportunities = generate_insights(insight_agent, processed_data, num_insights=5)
                    else:
                        # If the method doesn't accept num_insights, we need a workaround
                        # We'll modify the prompt passed to the agent to request 5 insights
                        original_prompt = getattr(insight_agent, 'prompt_template', None)
                        
                        if original_prompt and hasattr(insight_agent, 'prompt_template'):
                            # Temporarily modify the prompt template to request 5 insights
                            if "Generate insights" in insight_agent.prompt_template:
                                original_prompt = insight_agent.prompt_template
                                insight_agent.prompt_template = insight_agent.prompt_template.replace(
                                    "Generate insights", 
                                    "Generate exactly 5 insights"
                                )
                        
                        # Generate the insights
                        opportunities = generate_insights(insight_agent, processed_data)
                        
                        # Restore the original prompt if it was modified
                        if original_prompt and hasattr(insight_agent, 'prompt_template'):
                            insight_agent.prompt_template = original_prompt
                    
                    # If we still don't have 5 insights, pad the list
                    if len(opportunities) < 5:
                        st.warning(f"Only {len(opportunities)} insights were generated. Some placeholders will be used.")
                        
                        # Pad with placeholder opportunities
                        while len(opportunities) < 5:
                            idx = len(opportunities) + 1
                            opportunities.append({
                                'title': f"Potential Opportunity {idx}",
                                'description': "Insufficient data to generate this opportunity automatically. Consider manually reviewing tickets for additional insights.",
                                'issue': "Not enough data for automated analysis",
                                'solution': "Manual review recommended",
                                'justification': "AI requires more diverse ticket data to identify additional patterns",
                                'impact': "Unknown - requires human analysis"
                            })
                    
                    st.session_state['tab2_opportunities'] = opportunities
                except Exception as e:
                    st.error(f"Error generating opportunities: {e}")
                    st.session_state['tab2_opportunities'] = None
                    # Add rate limit handling
                    if "429" in str(e):
                        st.warning("API rate limit reached. Please wait a moment before trying again.")
                        # Add retry logic with exponential backoff
                        st.info("Retrying in 10 seconds...")
                        time.sleep(10)
                        st.experimental_rerun()
    
    # Display opportunities if they have been generated
    if st.session_state.get('tab2_insights_shown', False):
        opportunities = st.session_state.get('tab2_opportunities')
        
        if opportunities:
            # Limit number of opportunities to display (show exactly 5)
            max_opportunities = min(5, len(opportunities))
            displayed_opportunities = opportunities[:max_opportunities]
            
            # Display all opportunities as structured expandable blocks
            for i, opportunity in enumerate(displayed_opportunities):
                with st.expander(f"Opportunity {i+1}: {opportunity['title']}", expanded=True):
                    # Display structured information
                    st.markdown("### Issue Identified")
                    if 'issue' in opportunity:
                        st.markdown(opportunity['issue'])
                    else:
                        # Extract issue from description if not available as separate field
                        st.markdown(opportunity['description'].split('\n')[0] if '\n' in opportunity['description'] else opportunity['description'])
                    
                    st.markdown("### Proposed Solution")
                    if 'solution' in opportunity:
                        st.markdown(opportunity['solution'])
                    else:
                        # Extract solution if not available
                        parts = opportunity['description'].split('\n')
                        st.markdown(parts[1] if len(parts) > 1 else "Solution details included in implementation plan")
                    
                    st.markdown("### Justification")
                    if 'justification' in opportunity:
                        st.markdown(opportunity['justification'])
                    else:
                        # Extract justification if not available
                        parts = opportunity['description'].split('\n')
                        st.markdown(parts[2] if len(parts) > 2 else "See implementation plan for full justification")
                    
                    st.markdown("### Automation Impact")
                    if 'impact' in opportunity:
                        st.markdown(opportunity['impact'])
                    else:
                        # Extract impact if not available
                        parts = opportunity['description'].split('\n')
                        st.markdown(parts[3] if len(parts) > 3 else "Impact assessment included in implementation plan")
                    
                    # Implementation plan with unique key
                    plan_key = f"tab2_impl_plan_shown_{i}"
                    if plan_key not in st.session_state:
                        st.session_state[plan_key] = False
                        
                    if not st.session_state[plan_key]:
                        if st.button("Get Implementation Plan", key=f"btn_tab2_impl_plan_{i}"):
                            st.session_state[plan_key] = True
                            
                            with st.spinner("Generating implementation plan..."):
                                try:
                                    # Generate plan for this opportunity
                                    plan = generate_plan_for_opportunity(implementation_agent, opportunity)
                                    st.session_state[f'tab2_impl_plan_{i}'] = plan
                                except Exception as e:
                                    st.error(f"Error generating implementation plan: {e}")
                                    st.session_state[f'tab2_impl_plan_{i}'] = None
                    
                    # Display implementation plan if it has been generated
                    if st.session_state.get(plan_key, False):
                        impl_plan = st.session_state.get(f'tab2_impl_plan_{i}')
                        if impl_plan:
                            st.markdown("### Implementation Plan")
                            st.markdown(impl_plan)
        
        # Add reset button
        if st.button("Reset Insights", key="btn_reset_tab2"):
            # Clear all tab2 state variables
            tab2_keys = [key for key in st.session_state.keys() if key.startswith('tab2_')]
            for key in tab2_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['tab2_insights_shown'] = False
            st.experimental_rerun()

def custom_query_tab(processed_data, query_agent):
    """
    Tab for custom user queries
    
    Args:
        processed_data (pd.DataFrame): Processed ticket data
        query_agent (UserQueryAgent): Agent for handling queries
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
                                query_agent,
                                current_query, 
                                processed_data
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
    
    # File upload
    processed_data, keywords = handle_file_upload(data_agent)
    
    # Tabs for different functionalities
    if processed_data is not None:
        # Initialize active tab index if not present
        if 'tab_index' not in st.session_state:
            st.session_state['tab_index'] = 0

        # Check if tab has changed
        previous_tab = st.session_state.get('tab_index', 0)

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
        if previous_tab != selected_tab:
            # Reset states for all tabs when switching
            tab_state_keys = [
                key for key in st.session_state.keys() 
                if key.startswith('tab1_') or key.startswith('tab2_') or key.startswith('tab3_')
            ]
            for key in tab_state_keys:
                if key.endswith('_shown'):
                    st.session_state[key] = False
        
        # Show the selected tab content
        if selected_tab == 0:
            predefined_questions_tab(processed_data, query_agent)
        elif selected_tab == 1:
            automation_opportunities_tab(processed_data, insight_agent, implementation_agent)
        else:
            custom_query_tab(processed_data, query_agent)
    else:
        st.info("Please upload ticket data to begin analysis")

if __name__ == "__main__":
    main()