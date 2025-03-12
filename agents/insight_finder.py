import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import re
from collections import Counter, defaultdict
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

class InsightFinderAgent:
    """
    Agent 2: Responsible for identifying patterns in the processed data, 
    finding similar keywords in closed notes, and suggesting automation opportunities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_data = None
        self.text_data = None
        self.patterns = None
        self.automation_suggestions = None
        
        # Initialize NLP tools
        try:
            # Download required NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Ensure punkt is properly downloaded
            if not nltk.data.find('tokenizers/punkt'):
                nltk.download('punkt', download_dir=nltk.data.path[0])
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            self.logger.warning(f"Error initializing NLTK tools: {str(e)}")
            # Create a simple set of stop words as fallback
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                               'at', 'from', 'in', 'on', 'for', 'with', 'by', 'to', 'of', 'is', 'was',
                               'were', 'be', 'been', 'being', 'am', 'are', 'this', 'that', 'these', 'those'}
            self.lemmatizer = None
        
        # Initialize LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> Optional[ChatGroq]:
        """Initialize the LLM for automation suggestions."""
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
    
    def process_data(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data package from Agent 1.
        
        Args:
            data_package: Data package from Agent 1
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Processing data from Agent 1")
        
        # Store the processed data
        self.processed_data = data_package.get("processed_data")
        if self.processed_data is None:
            self.logger.error("No processed data found in the data package")
            return {"error": "No processed data found"}
        
        # Extract all relevant text fields
        self.text_data = data_package.get("text_columns", {})
        
        # Create comprehensive ticket representations for better analysis
        self.comprehensive_tickets = []
        
        # Determine how many tickets we have from the first available field
        if self.text_data:
            first_field = next(iter(self.text_data.values()))
            num_tickets = len(first_field)
            
            for i in range(num_tickets):
                ticket_info = {}
                # Collect all available fields for this ticket
                for field_name, field_data in self.text_data.items():
                    if i < len(field_data) and field_data[i]:
                        ticket_info[field_name] = field_data[i]
                
                if ticket_info:
                    self.comprehensive_tickets.append(ticket_info)
        
        # Check if we have required fields
        required_columns = ["close_notes", "resolution"]
        for column in required_columns:
            if column not in self.text_data:
                self.logger.warning(f"'{column}' not found in data - pattern detection may be less accurate")
        
        # Log the number of comprehensive tickets created
        self.logger.info(f"Created {len(self.comprehensive_tickets)} comprehensive ticket representations")
        
        # Categorize tickets into predefined categories
        self._categorize_tickets()
        self.logger.info(f"Categorized tickets into {len(self.patterns)} patterns")
        
        # Generate automation suggestions
        self.automation_suggestions = self._generate_automation_suggestions()
        self.logger.info(f"Generated {len(self.automation_suggestions)} automation suggestions")
        
        # Generate structured report
        report = self.generate_structured_report()
        
        return {
            "patterns": self.patterns,
            "automation_suggestions": self.automation_suggestions,
            "structured_report": report
        }
    
    def _preprocess_text(self, text_list: List[str]) -> List[str]:
        """
        Preprocess text data for analysis.
        
        Args:
            text_list: List of text strings
            
        Returns:
            List of preprocessed text strings
        """
        processed_texts = []
        
        try:
            for text in text_list:
                if not text or text == 'nan':
                    processed_texts.append('')
                    continue
                    
                # Convert to lowercase
                text = text.lower()
                
                # Remove special characters and digits
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                
                # Try to use NLTK tokenization, fall back to simple split if unavailable
                try:
                    from nltk.tokenize import word_tokenize
                    tokens = word_tokenize(text) if self.lemmatizer else text.split()
                except (ImportError, LookupError):
                    self.logger.warning("NLTK tokenization unavailable, using basic split")
                    tokens = text.split()
                
                # Remove stop words and lemmatize
                if self.lemmatizer:
                    try:
                        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                                         if token not in self.stop_words and len(token) > 2]
                    except Exception as e:
                        self.logger.warning(f"Lemmatization failed: {str(e)}, using basic filtering")
                        filtered_tokens = [token for token in tokens 
                                         if token not in self.stop_words and len(token) > 2]
                else:
                    filtered_tokens = [token for token in tokens 
                                     if token not in self.stop_words and len(token) > 2]
                
                # Join tokens back to string
                processed_text = ' '.join(filtered_tokens)
                processed_texts.append(processed_text)
        
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {str(e)}")
            # Fallback to basic preprocessing
            for text in text_list:
                if not text or text == 'nan':
                    processed_texts.append('')
                    continue
                
                # Simple preprocessing
                text = text.lower()
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                words = [w for w in text.split() if len(w) > 2 and w not in self.stop_words]
                processed_texts.append(' '.join(words))
        
        return processed_texts
    
    def _identify_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify patterns in the text data.
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Focus on close notes if available
        if "close_notes" in self.text_data and self.text_data["close_notes"]:
            notes = self.text_data["close_notes"]
            
            try:
                # Try the advanced NLP approach first
                self.logger.info("Attempting pattern identification with advanced NLP")
                
                # Preprocess the text
                processed_notes = self._preprocess_text(notes)
                
                # Remove empty strings
                non_empty_notes = [note for note in processed_notes if note.strip()]
                
                if len(non_empty_notes) > 10:  # Only analyze if we have enough data
                    # Use TF-IDF vectorization to capture important terms
                    vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.7)
                    
                    tfidf_matrix = vectorizer.fit_transform(non_empty_notes)
                    
                    # Use KMeans to cluster similar notes
                    num_clusters = min(5, len(non_empty_notes) // 5)  # Limit clusters to reasonable number
                    num_clusters = max(2, num_clusters)  # At least 2 clusters
                    
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    kmeans.fit(tfidf_matrix)
                    
                    # Get cluster assignments
                    clusters = kmeans.labels_
                    
                    # Extract features (keywords) for each cluster
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get cluster centers
                    cluster_centers = kmeans.cluster_centers_
                    
                    # For each cluster, find the top keywords
                    for i in range(num_clusters):
                        # Get indices of notes in this cluster
                        cluster_indices = [idx for idx, label in enumerate(clusters) if label == i]
                        
                        # Get original notes in this cluster
                        cluster_notes = [notes[idx] for idx in cluster_indices]
                        
                        # Get top terms for this cluster
                        cluster_center = cluster_centers[i]
                        top_term_indices = cluster_center.argsort()[-10:][::-1]
                        top_terms = [feature_names[idx] for idx in top_term_indices]
                        
                        # Create pattern dictionary
                        pattern = {
                            "id": f"pattern_{i+1}",
                            "keywords": top_terms,
                            "frequency": len(cluster_indices),
                            "sample_notes": cluster_notes[:5],  # Include a few sample notes
                            "percentage": len(cluster_indices) / len(notes) * 100
                        }
                        
                        patterns.append(pattern)
                
            except Exception as e:
                self.logger.error(f"Error in advanced NLP pattern detection: {str(e)}")
                self.logger.info("Falling back to simple pattern detection")
                patterns = []  # Reset patterns to use simpler approach
        
        # If we have no patterns yet, use a simple frequency-based approach
        if not patterns and "close_notes" in self.text_data:
            self.logger.info("Using fallback frequency-based pattern detection")
            try:
                # Simple approach - frequency-based pattern detection
                notes = self.text_data["close_notes"]
                
                # Initialize counters for single words and phrases
                word_counter = Counter()
                phrase_counter = Counter()
                
                # Process each note
                for note in notes:
                    if not note or note == 'nan':
                        continue
                        
                    # Convert to lowercase and clean text
                    text = note.lower()
                    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                    
                    # Count single words
                    words = [w for w in text.split() if len(w) > 3 and w not in self.stop_words]
                    word_counter.update(words)
                    
                    # Count 2-3 word phrases
                    phrases = re.findall(r'\b(\w+\s+\w+(?:\s+\w+)?)\b', text)
                    phrase_counter.update(phrases)
                
                # Create patterns from common words
                top_words = word_counter.most_common(10)
                for i, (word, count) in enumerate(top_words):
                    if count > 2:  # Only include if it appears multiple times
                        # Find sample notes containing this word
                        sample_notes = [note for note in notes if word in note.lower()]
                        
                        pattern = {
                            "id": f"pattern_word_{i+1}",
                            "keywords": [word],
                            "frequency": count,
                            "sample_notes": sample_notes[:5],
                            "percentage": count / len(notes) * 100
                        }
                        
                        patterns.append(pattern)
                
                # Create patterns from common phrases
                top_phrases = phrase_counter.most_common(5)
                for i, (phrase, count) in enumerate(top_phrases):
                    if count > 1:  # Only include if it appears multiple times
                        # Find sample notes containing this phrase
                        sample_notes = [note for note in notes if phrase in note.lower()]
                        
                        pattern = {
                            "id": f"pattern_phrase_{i+1}",
                            "keywords": [phrase],
                            "frequency": count,
                            "sample_notes": sample_notes[:5],
                            "percentage": count / len(notes) * 100
                        }
                        
                        patterns.append(pattern)
                
            except Exception as e:
                self.logger.error(f"Error in simple pattern detection: {str(e)}")
        
        # If we still have no patterns, create some dummy patterns
        if not patterns:
            self.logger.warning("Unable to detect patterns, creating placeholders")
            
            # Create some placeholder patterns
            pattern = {
                "id": "pattern_default_1",
                "keywords": ["general", "support", "issue"],
                "frequency": 0,
                "sample_notes": [],
                "percentage": 0
            }
            patterns.append(pattern)
        
        return patterns
    
    def _generate_automation_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate automation suggestions based on identified patterns.
        Suggestions shall not be blank, if no suggestion is identified then mention "No suggestion identified"
        
        Returns:
            List of automation suggestion dictionaries
        """
        suggestions = []
        
        # Check if we have both patterns and LLM initialized
        if not self.patterns or not self.llm:
            self.logger.warning("Cannot generate automation suggestions without patterns or LLM")
            return suggestions
        
        # Create a prompt template for automation suggestions
        prompt_template = PromptTemplate(
            input_variables=["pattern"],
            template="""
            You are an experienced Automation Consultant. Analyze the following pattern from IT service desk tickets and suggest automation opportunities:
            
            Pattern Information:
            {pattern}
            
            Provide a crisp automation suggestion that includes:
            1. The root cause of the problem (what's causing these tickets)
            2. A specific suggested automation solution
            3. Why this is the best solution
            
            Keep your response concise, specific, and focused on practical automation solutions.
            """
        )
        
        # Create an LLM chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # Process each pattern
        for pattern in self.patterns:
            self.logger.info(f"Generating automation suggestion for pattern {pattern['id']}")
            
            try:
                # Prepare pattern info for the LLM
                pattern_info = (
                    f"Keywords: {', '.join(pattern['keywords'])}\n"
                    f"Frequency: {pattern['frequency']} occurrences ({pattern['percentage']:.1f}% of tickets)\n"
                    f"Sample Ticket Notes:\n" + "\n".join([f"- {note[:200]}..." if len(note) > 200 else f"- {note}" 
                                                         for note in pattern['sample_notes'][:3]])
                )
                
                # Get the suggestion from the LLM
                response = chain.run({"pattern": pattern_info})
                
                # Parse the response to extract structured information
                root_cause = ""
                solution = ""
                justification = ""
                
                if "root cause" in response.lower():
                    parts = response.split("\n")
                    for i, part in enumerate(parts):
                        if "root cause" in part.lower():
                            root_cause = part.split(":", 1)[1].strip() if ":" in part else part.strip()
                        elif "suggested solution" in part.lower() or "automation solution" in part.lower():
                            solution = part.split(":", 1)[1].strip() if ":" in part else part.strip()
                        elif "best solution" in part.lower() or "why this is" in part.lower():
                            justification = part.split(":", 1)[1].strip() if ":" in part else part.strip()
                
                # If parsing failed, use the whole response
                if not root_cause and not solution:
                    solution = response
                
                # Create suggestion dictionary
                suggestion = {
                    "id": f"suggestion_{pattern['id']}",
                    "pattern_id": pattern['id'],
                    "keywords": pattern['keywords'],
                    "problem_root_cause": root_cause,
                    "suggested_solution": solution,
                    "justification": justification,
                    "frequency": pattern['frequency'],
                    "percentage": pattern['percentage']
                }
                
                suggestions.append(suggestion)
                
            except Exception as e:
                self.logger.error(f"Error generating suggestion for pattern {pattern['id']}: {str(e)}")
        
        return suggestions
    
    def generate_structured_report(self) -> Dict[str, Any]:
        """
        Generate a structured report of findings and automation opportunities.
        
        Returns:
            Dictionary containing structured report data
        """
        report = {
            "summary": {
                "total_tickets": len(self.comprehensive_tickets) if hasattr(self, 'comprehensive_tickets') else 0,
                "categories_identified": len(self.patterns) if self.patterns else 0,
                "automation_opportunities": len(self.automation_suggestions) if self.automation_suggestions else 0
            },
            "top_categories": [],
            "automation_opportunities": []
        }
        
        # Add top categories
        if self.patterns:
            for pattern in self.patterns[:5]:  # Top 5 categories
                report["top_categories"].append({
                    "name": pattern.get("category", "Unknown Category"),
                    "frequency": pattern.get("frequency", 0),
                    "percentage": pattern.get("percentage", 0),
                    "keywords": pattern.get("keywords", [])
                })
        
        # Add automation opportunities
        if self.automation_suggestions:
            for suggestion in self.automation_suggestions:
                report["automation_opportunities"].append({
                    "name": suggestion.get("opportunity_name", "Unknown Opportunity"),
                    "category": suggestion.get("category", "General"),
                    "frequency": suggestion.get("frequency", 0),
                    "root_cause": suggestion.get("problem_root_cause", "Unknown"),
                    "solution": suggestion.get("suggested_solution", "No solution available"),
                    "implementation": suggestion.get("implementation_steps", ""),
                    "benefits": suggestion.get("expected_benefits", "")
                })
        
        return report


    def _initialize_categories(self):
        """Initialize predefined incident categories for classification."""
        self.categories = {
            "password_reset": {
                "keywords": ["password", "reset", "forgot", "locked", "account lock", "credentials"],
                "count": 0,
                "examples": []
            },
            "application_errors": {
                "keywords": ["error", "crash", "bug", "not working", "failed", "exception"],
                "count": 0,
                "examples": []
            },
            "system_updates": {
                "keywords": ["update", "upgrade", "patch", "version", "install update"],
                "count": 0,
                "examples": []
            },
            "email_problems": {
                "keywords": ["email", "outlook", "message", "mailbox", "cannot send", "not receiving"],
                "count": 0,
                "examples": []
            },
            "access_requests": {
                "keywords": ["access", "permission", "authorize", "rights", "grant access"],
                "count": 0,
                "examples": []
            },
            "network_connectivity": {
                "keywords": ["network", "connect", "wifi", "internet", "vpn", "connection"],
                "count": 0,
                "examples": []
            },
            "software_installation": {
                "keywords": ["install", "setup", "download", "deploy", "software"],
                "count": 0,
                "examples": []
            },
            "account_management": {
                "keywords": ["account", "user", "profile", "create account", "disable", "enable"],
                "count": 0,
                "examples": []
            }
        }



    def _categorize_tickets(self):
        """Categorize tickets into predefined categories based on text content."""
        self._initialize_categories()
        
        for ticket in self.comprehensive_tickets:
            # Combine all text fields for this ticket
            combined_text = " ".join([text for field, text in ticket.items()])
            combined_text = combined_text.lower()
            
            # Check each category
            for category_name, category_info in self.categories.items():
                # Check if any keywords match
                if any(keyword in combined_text for keyword in category_info["keywords"]):
                    self.categories[category_name]["count"] += 1
                    
                    # Add as an example if we don't have too many yet
                    if len(self.categories[category_name]["examples"]) < 5:
                        # Create a summary of this ticket
                        ticket_summary = {
                            field: text[:100] + ("..." if len(text) > 100 else "")
                            for field, text in ticket.items()
                        }
                        self.categories[category_name]["examples"].append(ticket_summary)
        
        # Convert categories to patterns
        self.patterns = []
        
        # Sort categories by count to prioritize
        sorted_categories = sorted(
            self.categories.items(), 
            key=lambda x: x[1]["count"], 
            reverse=True
        )
        
        # Create patterns from top categories
        for i, (category_name, category_info) in enumerate(sorted_categories):
            if category_info["count"] > 0:
                pattern = {
                    "id": f"pattern_{i+1}",
                    "category": category_name.replace("_", " ").title(),
                    "keywords": category_info["keywords"],
                    "frequency": category_info["count"],
                    "sample_notes": [
                        str(example) for example in category_info["examples"][:3]
                    ],
                    "percentage": (category_info["count"] / len(self.comprehensive_tickets)) * 100
                    if self.comprehensive_tickets else 0
                }
                self.patterns.append(pattern)


    def get_top_suggestions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top automation suggestions.
        
        Args:
            limit: Maximum number of suggestions to return
            
        Returns:
            List of top automation suggestions
        """
        if not self.automation_suggestions:
            return []
        
        # Sort by frequency/impact
        sorted_suggestions = sorted(
            self.automation_suggestions, 
            key=lambda x: x['frequency'], 
            reverse=True
        )
        
        return sorted_suggestions[:limit]