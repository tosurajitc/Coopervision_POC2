import os
import re
from .groq_client import create_groq_client, call_groq_api
from rate_limit_handler import apply_rate_limit_handling

@apply_rate_limit_handling
class ImplementationStrategyAgent:
    """
    Agent 3: Implementation Strategy
    - Generates step-by-step implementation plans for automation suggestions
    - Plans should be concise (within 200 words)
    """
    
    def __init__(self):
        # Initialize GROQ client
        self.client, self.model_name, self.error_message = create_groq_client()
        if self.error_message:
            print(f"Warning: {self.error_message}")
    
    def generate_plan(self, question, answer):
        """
        Generate an implementation plan based on a question and its answer
        
        Args:
            question (str): The original question
            answer (str): The answer/insight provided
            
        Returns:
            str: A step-by-step implementation plan
        """
        if not self.client:
            return self._get_fallback_plan(question, answer)
        
        try:
            # Prompt for the LLM
            prompt = f"""
            Create an implementation plan for an automation opportunity.
            
            FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

# Implementation Plan

## Objective
[Objective statement]

## Required Tools/Technologies
- [Tool 1]
- [Tool 2]
- [Tool 3]

## Implementation Phases
1. [Phase 1 with timeframe]
2. [Phase 2 with timeframe] 
3. [Phase 3 with timeframe]
4. [Phase 4 with timeframe]

## Success Metrics
- [Metric 1]
- [Metric 2]
- [Metric 3]

IMPORTANT: Do not include any thinking, explanations, or reasoning - only the implementation plan in the format above.

Question: "{question}"
Analysis: "{answer}"
            """
            
            # Call the GROQ API
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            if error:
                print(f"Error generating implementation plan: {error}")
                return self._get_fallback_plan(question, answer)
                
            # Clean up the response
            response_text = self._clean_response(response_text)
            
            return response_text
            
        except Exception as e:
            print(f"Error generating implementation plan: {str(e)}")
            return self._get_fallback_plan(question, answer)
    
    def generate_plan_for_opportunity(self, opportunity):
        """
        Generate an implementation plan for a specific automation opportunity
        
        Args:
            opportunity (dict): The automation opportunity with title and description
            
        Returns:
            str: A step-by-step implementation plan
        """
        title = opportunity.get('title', '')
        description = opportunity.get('description', '')
        
        if not self.client:
            return self._get_fallback_plan_for_opportunity(title, description)
        
        try:
            # Prompt for the LLM
            prompt = f"""
            Create an implementation plan for an automation opportunity.
            
            FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

# Implementation Plan

## Objective
[Objective statement]

## Required Tools/Technologies
- [Tool 1]
- [Tool 2]
- [Tool 3]

## Implementation Phases
1. [Phase 1 with timeframe]
2. [Phase 2 with timeframe] 
3. [Phase 3 with timeframe]
4. [Phase 4 with timeframe]

## Success Metrics
- [Metric 1]
- [Metric 2]
- [Metric 3]

IMPORTANT: Do not include any thinking, explanations, or reasoning - only the implementation plan in the format above.

Opportunity: "{title}"
Description: "{description}"
            """
            
            # Call the GROQ API
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            if error:
                print(f"Error generating implementation plan: {error}")
                return self._get_fallback_plan_for_opportunity(title, description)
                
            # Clean up the response
            response_text = self._clean_response(response_text)
            
            return response_text
            
        except Exception as e:
            print(f"Error generating implementation plan: {str(e)}")
            return self._get_fallback_plan_for_opportunity(title, description)
    
    def _clean_response(self, text):
        """
        Clean the response to remove thinking content and HTML tags
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = text.replace("<think>", "").replace("</think>", "")
        text = text.replace("<div", "").replace("</div>", "")
        
        # Look for implementation plan section
        plan_markers = [
            "# Implementation Plan",
            "## Implementation Plan",
            "Implementation Plan:"
        ]
        
        for marker in plan_markers:
            if marker in text:
                text = text.split(marker, 1)[1]
                return "# Implementation Plan" + text
        
        # Remove thinking explanations
        thinking_phrases = [
            "Let me think about",
            "I'll create",
            "Based on the",
            "Looking at the",
            "To create an implementation plan",
            "Let's create",
            "First, I'll",
            "I need to",
            "To address this",
            "Let me analyze"
        ]
        
        for phrase in thinking_phrases:
            if text.startswith(phrase):
                # Find first markdown header and keep everything after it
                if "#" in text:
                    text = text.split("#", 1)[1]
                    text = "#" + text
        
        # Look for structured sections
        sections = ["Objective", "Required Tools", "Implementation Phases", "Success Metrics"]
        for section in sections:
            if "## " + section in text:
                # We have a properly formatted section, keep everything from the first heading
                return text
        
        # Fallback to default plan if we couldn't clean effectively
        return self._get_fallback_plan("", "")
    
    def _get_fallback_plan(self, question, answer):
        """
        Generate a fallback implementation plan when the LLM is not available
        
        Args:
            question (str): The original question
            answer (str): The answer/insight provided
            
        Returns:
            str: A generic implementation plan
        """
        return f"""# Implementation Plan

## Objective
Address the identified issue and implement automation for the ticket handling process.

## Required Tools/Technologies
- Workflow automation platform
- Integration tools
- Documentation system

## Implementation Phases
1. **Assessment (Week 1-2)**: Document current processes and validate the issue scope.
2. **Design (Week 3-4)**: Create solution design and get stakeholder approval.
3. **Development (Week 5-8)**: Build and test the automation solution.
4. **Deployment (Week 9-10)**: Roll out the solution with user training.

## Success Metrics
- Reduction in related tickets
- Decreased resolution time
- Improved user satisfaction
        """
    
    def _get_fallback_plan_for_opportunity(self, title, description):
        """
        Generate a fallback implementation plan for an opportunity when the LLM is not available
        
        Args:
            title (str): The opportunity title
            description (str): The opportunity description
            
        Returns:
            str: A generic implementation plan
        """
        return f"""# Implementation Plan

## Objective
Implement automation solution for the identified opportunity.

## Required Tools/Technologies
- Workflow automation platform
- Integration tools
- Documentation system

## Implementation Phases
1. **Assessment (Week 1-2)**: Document current processes and validate the opportunity scope.
2. **Design (Week 3-4)**: Create solution design and get stakeholder approval.
3. **Development (Week 5-8)**: Build and test the automation solution.
4. **Deployment (Week 9-10)**: Roll out the solution with user training.

## Success Metrics
- Reduction in manual effort
- Increased process efficiency
- Improved user satisfaction
        """