import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

class UserQueryAgent:
    """
    Agent 3: Responds to user queries about ticket data, provides insights,
    and suggests automation opportunities based on user questions.
    """
    
    # Standard questions with descriptions
    STANDARD_QUESTIONS = [
        {
            "id": "common_pain_points",
            "question": "What are the most common pain points reported by users in the ticket descriptions?",
            "description": "Understanding recurring frustrations can help pinpoint areas for automation."
        },
        {
            "id": "repetitive_steps",
            "question": "Which ticket resolutions involve repetitive manual steps that could be automated?",
            "description": "Identify resolution steps that follow a standard process and can be scripted."
        },
        {
            "id": "self_service",
            "question": "Are there tickets that could be resolved without human intervention if users had better self-service options?",
            "description": "Explore possibilities for chatbots, self-service portals, or knowledge base improvements."
        },
        {
            "id": "dependencies",
            "question": "Which types of issues take longer to resolve due to dependencies on other teams or approvals?",
            "description": "Find areas where automation can speed up approvals or cross-team coordination."
        },
        {
            "id": "misrouted_tickets",
            "question": "Are there common scenarios where misrouted or reassigned tickets delay resolution?",
            "description": "Investigate if automation can improve ticket categorization and assignment."
        },
        {
            "id": "proactive_prevention",
            "question": "Do users frequently report similar issues that could be proactively prevented?",
            "description": "Consider monitoring systems or predictive maintenance to reduce ticket creation."
        },
        {
            "id": "manual_data_entry",
            "question": "Which types of tickets require extensive manual data entry or retrieval?",
            "description": "Identify opportunities for automation through integrations or RPA solutions."
        },
        {
            "id": "communication_gaps",
            "question": "Are there communication gaps between users and support teams that lead to delays?",
            "description": "Look for opportunities to automate status updates, notifications, or reminders."
        },
        {
            "id": "workarounds",
            "question": "Do resolution notes indicate workarounds that could be standardized into automated processes?",
            "description": "Determine if repetitive troubleshooting steps could be embedded into AI-driven diagnostics."
        },
        {
            "id": "training_gaps",
            "question": "Are there tickets that result from lack of training or unclear processes?",
            "description": "Consider automation for guided workflows, interactive documentation, or training bots."
        }
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_processor_results = None
        self.insight_finder_results = None
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> Optional[ChatGroq]:
        """Initialize the LLM for responding to user queries."""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            model_name = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")
            
            if not api_key:
                self.logger.error("GROQ_API_KEY not found in environment variables")
                return None
                
            llm = ChatGroq(
                groq_api_key=api_key, 
                model_name=model_name
            )
            return llm
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {str(e)}")
            return None
    
    def initialize_with_data(self, data_processor_results: Dict[str, Any], insight_finder_results: Dict[str, Any]) -> None:
        """
        Initialize the agent with results from the other agents.
        
        Args:
            data_processor_results: Results from DataProcessorAgent
            insight_finder_results: Results from InsightFinderAgent
        """
        self.data_processor_results = data_processor_results
        self.insight_finder_results = insight_finder_results
        self.logger.info("UserQueryAgent initialized with data from other agents")
    
    def get_standard_questions(self) -> List[Dict[str, str]]:
        """
        Get the list of standard questions.
        
        Returns:
            List of standard question dictionaries
        """
        return self.STANDARD_QUESTIONS
    
    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response to remove any thinking process or headers.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response
        """
        # Handle None or empty responses
        if not response:
            return ""
            
        clean_response = response
        
        # Remove thinking process if present
        if "<think>" in clean_response:
            parts = clean_response.split("<think>")
            if len(parts) >= 2:
                # Get the text before the thinking part
                before_think = parts[0].strip()
                if before_think:
                    clean_response = before_think
                elif "</think>" in clean_response:
                    # If there's nothing before <think>, get what's after </think>
                    parts = clean_response.split("</think>")
                    if len(parts) >= 2:
                        clean_response = parts[1].strip()
        
        # Remove "Response:" prefix if present
        if clean_response.startswith("Response:"):
            clean_response = clean_response[9:].strip()
        
        return clean_response
    
    def process_query(self, query: str, is_standard: bool = False, question_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query text
            is_standard: Whether this is a standard question
            question_id: ID of the standard question if applicable
            
        Returns:
            Dictionary containing the response
        """
        self.logger.info(f"Processing query: {query}")
        
        if not self.llm:
            self.logger.error("LLM not initialized, cannot process query")
            return {
                "success": False,
                "error": "AI assistant not available",
                "response": "Unable to process your query at this time."
            }
        
        if not self.data_processor_results or not self.insight_finder_results:
            self.logger.error("Agent not initialized with data, cannot process query")
            return {
                "success": False,
                "error": "No ticket data loaded",
                "response": "Please upload ticket data before asking questions."
            }
        
        try:
            # Get the relevant data for the LLM
            summary_stats = self.data_processor_results.get("summary_stats", {})
            patterns = self.insight_finder_results.get("patterns", [])
            suggestions = self.insight_finder_results.get("automation_suggestions", [])
            
            # Prepare data summaries for the prompt
            stats_summary = self._format_summary_stats(summary_stats)
            patterns_summary = self._format_patterns(patterns)
            suggestions_summary = self._format_suggestions(suggestions)
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["query", "stats_summary", "patterns_summary", "suggestions_summary"],
                template="""
                You are an experienced data analyst specializing in IT service desk ticket analysis. 
                You need to answer the following question about the ticket data:
                
                USER QUESTION: {query}
                
                Here's a summary of the ticket data:
                {stats_summary}
                
                Here are the patterns identified in the ticket data:
                {patterns_summary}
                
                Here are the automation suggestions derived from these patterns:
                {suggestions_summary}
                
                Provide a concise, insightful response that:
                1. Directly addresses the user's question
                2. Identifies key insights related to their query
                3. Suggests specific automation opportunities relevant to their question
                
                Structure your response with at most 2 bullet points for each section (insights and automation suggestions).
                Be specific, practical, and focused on actionable information.
                Limit your response to 250 words maximum.
                
                IMPORTANT: DO NOT include any markers like "Response:" or any thinking process. Just provide the direct response without any preamble.
                DO NOT include "<think>" or any thinking process in your response.
                """
            )
            
            # Create an LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            # Get the response from the LLM
            response = chain.run({
                "query": query,
                "stats_summary": stats_summary,
                "patterns_summary": patterns_summary,
                "suggestions_summary": suggestions_summary
            })
            
            # Clean the response
            cleaned_response = self._clean_response(response)
            
            return {
                "success": True,
                "question": query,
                "is_standard": is_standard,
                "question_id": question_id,
                "response": cleaned_response
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "An error occurred while processing your query."
            }
    
    def _format_summary_stats(self, stats: Dict[str, Any]) -> str:
        """Format summary statistics for the prompt."""
        if not stats:
            return "No summary statistics available."
        
        lines = []
        
        # Basic counts
        if "total_tickets" in stats:
            lines.append(f"Total Tickets: {stats['total_tickets']}")
        
        # Time statistics
        if "avg_resolution_time" in stats:
            lines.append(f"Average Resolution Time: {stats['avg_resolution_time']:.2f} hours")
        
        if "resolution_band_distribution" in stats:
            lines.append("Resolution Time Distribution:")
            for band, count in stats["resolution_band_distribution"].items():
                lines.append(f"  - {band}: {count} tickets")
        
        # Assignment groups
        if "assignment_group_distribution" in stats:
            lines.append("Top Assignment Groups:")
            for group, count in list(stats["assignment_group_distribution"].items())[:5]:
                lines.append(f"  - {group}: {count} tickets")
        
        # Complexity
        if "complexity_distribution" in stats:
            lines.append("Ticket Complexity Distribution:")
            for complexity, count in stats["complexity_distribution"].items():
                lines.append(f"  - {complexity}: {count} tickets")
        
        # Common keywords
        if "common_keywords" in stats:
            lines.append("Common Keywords in Descriptions:")
            for word, count in list(stats["common_keywords"].items())[:10]:
                lines.append(f"  - '{word}': {count} occurrences")
        
        # Resolution keywords
        if "common_resolution_keywords" in stats:
            lines.append("Common Keywords in Resolutions:")
            for word, count in list(stats["common_resolution_keywords"].items())[:10]:
                lines.append(f"  - '{word}': {count} occurrences")
        
        return "\n".join(lines)
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for the prompt."""
        if not patterns:
            return "No patterns identified in the ticket data."
        
        lines = []
        
        for i, pattern in enumerate(patterns):
            lines.append(f"Pattern {i+1}:")
            lines.append(f"  - Keywords: {', '.join(pattern.get('keywords', []))}")
            lines.append(f"  - Frequency: {pattern.get('frequency', 0)} tickets ({pattern.get('percentage', 0):.1f}% of total)")
            lines.append("  - Sample Notes:")
            for note in pattern.get('sample_notes', [])[:2]:
                # Truncate long notes
                if len(note) > 100:
                    note = note[:100] + "..."
                lines.append(f"    * {note}")
        
        return "\n".join(lines)
    
    def _format_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format automation suggestions for the prompt."""
        if not suggestions:
            return "No automation suggestions available."
        
        lines = []
        
        for i, suggestion in enumerate(suggestions):
            lines.append(f"Suggestion {i+1}:")
            
            if "problem_root_cause" in suggestion and suggestion["problem_root_cause"]:
                lines.append(f"  - Problem Root Cause: {suggestion['problem_root_cause']}")
            
            if "suggested_solution" in suggestion and suggestion["suggested_solution"]:
                lines.append(f"  - Suggested Solution: {suggestion['suggested_solution']}")
            
            if "justification" in suggestion and suggestion["justification"]:
                lines.append(f"  - Justification: {suggestion['justification']}")
            
            lines.append(f"  - Applies to {suggestion.get('frequency', 0)} tickets ({suggestion.get('percentage', 0):.1f}% of total)")
        
        return "\n".join(lines)