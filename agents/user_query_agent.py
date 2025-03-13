import pandas as pd
import os
from collections import Counter
import re
from .groq_client import create_groq_client, call_groq_api
from rate_limit_handler import apply_rate_limit_handling

@apply_rate_limit_handling
class UserQueryAgent:
    """
    Agent for Providing Comprehensive Automation Solutions
    - Responds to user queries about ticket data
    - Offers detailed automation recommendations
    - Supports both general and specific implementation queries
    """
    
    def __init__(self):
        # Initialize GROQ client
        self.client, self.model_name, self.error_message = create_groq_client()
        if self.error_message:
            print(f"Warning: {self.error_message}")
        
        # Define predefined questions with their corresponding analysis methods
        self.predefined_questions = [
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
        
        # Store latest analysis results
        self.latest_analysis = {}
    
    def _get_implementation_plan(self, question_index):
        """
        Generate an implementation plan for a specific question
        
        Args:
            question_index (int): Index of the predefined question
            
        Returns:
            str: Detailed implementation plan
        """
        # Check if previous analysis exists
        if question_index not in self.latest_analysis:
            return "No previous analysis found. Please run the analysis first."
        
        # Prepare the implementation plan request
        previous_analysis = self.latest_analysis[question_index]
        
        if not self.client:
            return "Implementation plan generation requires an LLM client."
        
        try:
            # Prepare prompt for implementation plan
            prompt = f"""
            You are an expert IT automation consultant creating a comprehensive implementation plan.
            
            Previous Analysis:
            {previous_analysis}
            
            Based on the analysis above, provide a detailed implementation plan that includes:
            1. Specific steps to address the identified automation opportunity
            2. Technical and process-level recommendations
            3. Potential tools or technologies to use
            4. Expected benefits and ROI
            5. Potential challenges and mitigation strategies
            6. Phased approach for implementation
            
            Format your response with clear headings and actionable steps.
            """
            
            # Call GROQ API for implementation plan
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=1000,
                temperature=0.2
            )
            
            if error:
                return f"Error generating implementation plan: {error}"
            
            return response_text
        
        except Exception as e:
            return f"Error generating implementation plan: {str(e)}"

    def _generate_custom_solution(self, question, data):
        """
        Generate a custom solution for user's query
        
        Args:
            question (str): User's custom query
            data (pd.DataFrame): Processed ticket data
            
        Returns:
            str: Comprehensive solution with automation recommendations
        """
        if not self.client:
            return "Custom solution generation requires an LLM client."
        
        try:
            # Prepare a sample of the data for analysis
            sample_size = min(100, len(data))
            sample_data = data.sample(sample_size) if len(data) > sample_size else data
            
            # Convert sample data to a string representation
            data_str = sample_data.head(20).to_string()
            
            # Prepare prompt for custom solution
            prompt = f"""
            You are an expert IT support analyst providing a direct, actionable solution.

            Strictly follow these guidelines:
            - Provide a clear, concise response
            - DO NOT include any "Think:" or preparation steps
            - Directly answer the query with practical insights
            - Focus on actionable automation recommendations

            User Query: "{question}"

            Ticket Data Sample:
            {data_str}

            Response Format:
            ## Analysis
            [Brief analysis of the query]

            ## Automation Opportunities
            [Specific, actionable automation recommendations]

            ## Implementation Strategy
            [Practical steps to implement the recommendations]
            """
            
            # Call GROQ API for custom solution
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=1000,
                temperature=0.2
            )
            
            if error:
                return f"Error generating custom solution: {error}"
            
            # Remove any think blocks or preprocessing text
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            response_text = re.sub(r'Think:.*?(?=\n\n|\n|$)', '', response_text, flags=re.DOTALL)
            
            return response_text
        
        except Exception as e:
            return f"Error generating custom solution: {str(e)}"


    def answer_question(self, question, data):
        """
        Comprehensive query handling method
        
        Args:
            question (str): User's query
            data (pd.DataFrame): Processed ticket data
            
        Returns:
            str: Detailed analysis and automation recommendations
        """
        # Normalize the question
        question = question.lower().strip()
        
        # Check for implementation plan request
        implementation_match = re.search(r'implement(?:ation)?\s*(?:plan)?\s*(?:for)?\s*(?:question)?\s*(\d+)', question)
        if implementation_match:
            question_index = int(implementation_match.group(1)) - 1
            return self._get_implementation_plan(question_index)
        
        # Check if the question matches a predefined question
        for idx, predefined_q in enumerate(self.predefined_questions):
            if predefined_q.lower() in question:
                return self._analyze_predefined_question(idx, data)
        
        # If it's a custom query, use LLM for comprehensive analysis
        return self._generate_custom_solution(question, data)
    
    def _analyze_predefined_question(self, question_index, data):
        """
        Analyze a specific predefined question
        
        Args:
            question_index (int): Index of the predefined question
            data (pd.DataFrame): Processed ticket data
            
        Returns:
            str: Detailed analysis with automation recommendations
        """
        # Predefined analysis methods mapped to questions
        analysis_methods = [
            self._analyze_pain_points,
            self._analyze_repetitive_steps,
            self._analyze_self_service,
            self._analyze_dependencies,
            self._analyze_reassignments,
            self._analyze_preventable_issues,
            self._analyze_data_entry,
            self._analyze_communication_gaps,
            self._analyze_workarounds,
            self._analyze_training_issues
        ]
        
        try:
            # Perform analysis
            analysis_result = analysis_methods[question_index](data)
            
            # Store the latest analysis for potential implementation plan
            self.latest_analysis[question_index] = analysis_result
            
            return analysis_result
        except Exception as e:
            return f"Error analyzing the question: {str(e)}"
    
    def _analyze_pain_points(self, data):
        """Analyze most frequently reported pain points"""
        try:
            # Extract common terms from descriptions
            common_terms = self._extract_common_terms(data, 'description', 20)
            pain_point_keywords = ['error', 'issue', 'problem', 'broken', 'fail', 'crash', 
                                  'not working', 'slow', 'unable', 'difficulty', 'cannot', 
                                  'stuck', 'freeze', 'hang']
            
            # Filter for pain point related terms
            pain_points = {term: count for term, count in common_terms.items() 
                          if any(keyword in term.lower() for keyword in pain_point_keywords)}
            
            # Get top pain points
            top_pain_points = dict(sorted(pain_points.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Create the response
            response = "## Most Frequently Reported Pain Points\n\n"
            
            if top_pain_points:
                for term, count in top_pain_points.items():
                    percentage = (count / len(data)) * 100
                    response += f"- **{term}**: {count} tickets ({percentage:.1f}%)\n"
                
                # Add automation recommendations
                response += "\n## Automation Recommendations\n\n"
                response += "- Implement self-service solutions for the most common pain points to reduce ticket volume\n"
                response += "- Create automated diagnostics and resolution for recurring technical issues\n"
            else:
                response = "No clear pain points were identified in the ticket descriptions."
            
            return response
        except Exception as e:
            return f"Error analyzing pain points: {str(e)}"
    
    # Placeholder methods for other analyses
    def _analyze_repetitive_steps(self, data):
        return "## Repetitive Steps Analysis\nPlaceholder for repetitive steps analysis"
    
    def _analyze_self_service(self, data):
        return "## Self-Service Opportunities\nPlaceholder for self-service analysis"
    
    def _analyze_dependencies(self, data):
        return "## Dependencies Analysis\nPlaceholder for dependencies analysis"
    
    def _analyze_reassignments(self, data):
        return "## Reassignments Analysis\nPlaceholder for reassignments analysis"
    
    def _analyze_preventable_issues(self, data):
        return "## Preventable Issues\nPlaceholder for preventable issues analysis"
    
    def _analyze_data_entry(self, data):
        return "## Data Entry Analysis\nPlaceholder for data entry analysis"
    
    def _analyze_communication_gaps(self, data):
        return "## Communication Gaps\nPlaceholder for communication gaps analysis"
    
    def _analyze_workarounds(self, data):
        return "## Workarounds Analysis\nPlaceholder for workarounds analysis"
    
    def _analyze_training_issues(self, data):
        return "## Training Issues\nPlaceholder for training issues analysis"
    
    def _extract_common_terms(self, data, column, top_n=10):
        """
        Extract common terms from a text column
        
        Args:
            data (pd.DataFrame): The data to analyze
            column (str): The column name to analyze
            top_n (int): Number of top terms to return
            
        Returns:
            dict: Dictionary of terms and their frequencies
        """
        if column not in data.columns:
            return {}
        
        # Combine all text from the column
        all_text = ' '.join(data[column].fillna('').astype(str))
        
        # Convert to lowercase and split into words
        words = all_text.lower().split()
        
        # Remove common stopwords
        stopwords = ['the', 'and', 'is', 'in', 'to', 'a', 'for', 'of', 'with', 'on', 'by']
        
        filtered_words = [word for word in words if len(word) > 3 and word not in stopwords]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Sort by frequency and get top_n
        sorted_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return sorted_counts
    
    # Existing methods like _get_implementation_plan, _generate_custom_solution remain the same
    # (Copy these methods from the previous implementation)