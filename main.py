import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Entry point for running the Ticket Automation Advisor application.
    This function sets up the environment and launches the Streamlit UI.
    """
    logger.info("Starting Ticket Automation Advisor")
    
    # Load environment variables
    load_dotenv()
    
    # Check if GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not found in environment variables. LLM functionality will be limited.")
    
    # Run the Streamlit application
    logger.info("Launching Streamlit UI")
    os.system("streamlit run ui/streamlit_app.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)