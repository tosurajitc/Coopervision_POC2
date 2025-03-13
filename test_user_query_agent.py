import pandas as pd
import os
import sys
import logging
from dotenv import load_dotenv  # Correct spelling

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the UserQueryAgent
try:
    from agents.user_query_agent import UserQueryAgent
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

def create_sample_dataframe():
    """
    Create a sample DataFrame to simulate ticket data
    """
    try:
        data = {
            'ticket_id': range(1, 101),
            'description': [
                'Need password reset for my account',
                'Requesting access to new system',
                'Manual data entry for monthly report',
                'Server disk space running low',
                'Need to create a new user account',
                'Forgot password again',
                'Cannot access shared drive',
                'Need to generate weekly sales report',
                'System alert: CPU usage high',
                'Routine server maintenance required'
            ] * 10,
            'resolution': [
                'Reset password for user',
                'Granted access after approval',
                'Manually compiled report data',
                'Cleared temp files to free disk space',
                'Created new user account',
                'Password reset completed',
                'Updated access permissions',
                'Report generated manually',
                'Investigated and resolved CPU spike',
                'Performed scheduled maintenance'
            ] * 10,
            'duration_hours': [2, 4, 6, 1, 3] * 20,
            'reassignment_indicator': [False, True, False, True, False] * 20,
            'assignment_group': ['IT Support', 'Network', 'Data Team', 'Server Ops', 'HR'] * 20
        }
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error creating sample dataframe: {e}")
        raise

def test_user_query_agent():
    """
    Comprehensive test for UserQueryAgent
    """
    logger.info("🤖 Starting User Query Agent Test 🤖")
    
    try:
        # Create sample data
        sample_data = create_sample_dataframe()
        logger.info(f"Sample data created with {len(sample_data)} rows")
        
        # Initialize the agent
        agent = UserQueryAgent()
        logger.info("UserQueryAgent initialized")
        
        # Test predefined questions
        predefined_questions = [
            "What are the most frequently reported pain points in ticket descriptions?",
            "Which ticket resolutions involve repetitive manual steps that could be automated?",
            "Are there tickets that could be resolved without human intervention through better self-service options?",
            "Which issues experience delays due to dependencies on other teams or approval workflows?",
            "Are tickets often misrouted or reassigned, leading to resolution delays?",
            "Do users frequently report similar issues that could be proactively prevented?",
            "Which tickets involve extensive manual data entry or retrieval?",
            "Are there common communication gaps causing delays between users and support teams?",
            "Do resolution notes indicate recurring workarounds that could be automated?",
            "Are certain tickets caused by a lack of training or unclear processes?"
        ]
        
        logger.info("Starting predefined question tests")
        for i, question in enumerate(predefined_questions, 1):
            logger.info(f"\n📋 Testing Question {i}: {question}")
            try:
                # Analyze the question
                analysis = agent.answer_question(question, sample_data)
                print(f"\n🔍 Analysis for Question {i}:")
                print(analysis)
                
                # Test implementation plan
                print(f"\n🛠️ Implementation Plan for Question {i}:")
                implementation_plan = agent.answer_question(f"Implement plan for question {i}", sample_data)
                print(implementation_plan)
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                print(f"❌ Error processing question {i}: {e}")
        
        # Test custom queries
        logger.info("Starting custom query tests")
        custom_queries = [
            "How can we reduce ticket resolution time?",
            "What are the main causes of ticket escalations?",
            "Suggest ways to improve our support process"
        ]
        
        for query in custom_queries:
            logger.info(f"\n❓ Testing Custom Query: {query}")
            try:
                custom_solution = agent.answer_question(query, sample_data)
                print(f"\n❓ Custom Query: {query}")
                print("Custom Solution:")
                print(custom_solution)
            except Exception as e:
                logger.error(f"Error processing custom query: {e}")
                print(f"❌ Error processing custom query: {e}")
    
    except Exception as e:
        logger.error(f"Critical error in test_user_query_agent: {e}")
        print(f"❌ Critical error: {e}")

def main():
    # Ensure environment is set up
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        return
    
    # Run the test
    test_user_query_agent()

if __name__ == "__main__":
    main()