import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Import the InsightGenerationAgent
from agents.insight_generation_agent import InsightGenerationAgent

def create_sample_dataframe():
    """
    Create a sample DataFrame to simulate ticket data
    """
    data = {
        'ticket_id': range(1, 101),
        'description': [
            'Need password reset for my account',
            'Requesting access to new system',
            'Manual data entry for monthly report',
            'Server disk space running low',
            'Need to create a new user account',
            # Add more varied descriptions
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
            # Corresponding resolutions
            'Password reset completed',
            'Updated access permissions',
            'Report generated manually',
            'Investigated and resolved CPU spike',
            'Performed scheduled maintenance'
        ] * 10,
        'duration_hours': [2, 4, 6, 1, 3] * 20,
        'reassignment_indicator': [False, True, False, True, False] * 20
    }
    return pd.DataFrame(data)

def test_insight_generation_agent():
    """
    Comprehensive test for InsightGenerationAgent
    """
    print("Testing Insight Generation Agent:")
    
    # Create sample data
    sample_data = create_sample_dataframe()
    
    # Optional: Provide some keywords
    keywords = ['password', 'access', 'report', 'system']
    
    # Initialize the agent
    agent = InsightGenerationAgent()
    
    # Generate insights
    try:
        insights = agent.generate_insights(sample_data, keywords)
        
        print("\n🔍 Insights Generated:")
        for i, insight in enumerate(insights, 1):
            print(f"\nOpportunity {i}:")
            print(f"Title: {insight.get('title', 'N/A')}")
            print(f"Description: {insight.get('description', 'N/A')}")
            print(f"Category: {insight.get('category', 'N/A')}")
            print(f"Impact: {insight.get('impact', 'N/A')}")
            
            # Print sample tickets if available
            if 'sample_tickets' in insight:
                print(f"Sample Tickets: {insight['sample_tickets']}")
        
        # Validate insights
        assert len(insights) > 0, "No insights generated"
        print("\n✅ Insight Generation Successful")
    
    except Exception as e:
        print(f"❌ Error in Insight Generation: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Ensure environment is set up
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        return
    
    # Run the test
    test_insight_generation_agent()

if __name__ == "__main__":
    main()