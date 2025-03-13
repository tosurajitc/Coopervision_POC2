#!/usr/bin/env python
"""
Test script for the Implementation Strategy Agent.
This standalone script tests the agent's ability to generate 
implementation plans without <think> tags or thinking process content.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure the agents module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.implementation_strategy_agent import ImplementationStrategyAgent
from agents.groq_client import create_groq_client, call_groq_api

def main():
    """Test the implementation strategy agent."""
    # Load environment variables
    load_dotenv()
    
    # Display GROQ API key status
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL_NAME")
    if groq_api_key:
        print(f"GROQ API key: {'*' * (len(groq_api_key) - 4)}{groq_api_key[-4:]}")
        print(f"GROQ Model: {groq_model}")
    else:
        print("Warning: GROQ API key not found in environment")
        print("Please set the GROQ_API_KEY environment variable or add it to your .env file")
        return
    
    # Initialize the implementation strategy agent
    print("\nInitializing implementation strategy agent...")
    agent = ImplementationStrategyAgent()
    
    if agent.error_message:
        print(f"Failed to initialize agent: {agent.error_message}")
        return
    
    print("Agent initialized successfully!")
    
    # Test case 1: Simple question and answer
    print("\n=== TEST CASE 1: Simple Question and Answer ===")
    question = "Which ticket resolutions involve repetitive manual steps that could be automated?"
    answer = """
    Analysis shows that 35% of tickets involve repetitive manual steps that could be automated.
    The most common repetitive tasks include:
    1. Password resets (15% of tickets)
    2. Software installation requests (12% of tickets)
    3. Data entry and validation (8% of tickets)
    These tasks follow predictable patterns and have clear workflows, making them ideal candidates for automation.
    """
    
    print(f"Generating implementation plan for question: {question}")
    print(f"Based on answer: {answer[:100]}...")
    plan = agent.generate_plan(question, answer)
    
    print("\nRESULT:")
    print("-" * 80)
    print(plan)
    print("-" * 80)
    
    # Test case 2: Automation opportunity
    print("\n=== TEST CASE 2: Automation Opportunity ===")
    opportunity = {
        "title": "Automated Password Reset System",
        "description": "Implement a self-service password reset system to reduce manual intervention by IT staff. Currently, password reset requests account for 15% of all tickets and require manual verification and reset steps by support personnel."
    }
    
    print(f"Generating implementation plan for opportunity: {opportunity['title']}")
    print(f"Based on description: {opportunity['description'][:100]}...")
    plan = agent.generate_plan_for_opportunity(opportunity)
    
    print("\nRESULT:")
    print("-" * 80)
    print(plan)
    print("-" * 80)
    
    # Summarize results
    print("\nTest Summary:")
    print("- Check if the output is properly formatted with sections for Objective, Tools, Phases, and Metrics")
    print("- Verify that there are no <think> tags or thinking process content")
    print("- Confirm that the implementation plan is concise and actionable")

if __name__ == "__main__":
    main()