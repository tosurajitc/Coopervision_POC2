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
        
        # Extract text data
        self.text_data = data_package.get("text_columns", {})
        if not self.text_data:
            self.logger.warning("No text columns found in the data package")
        
        # Find patterns in the data
        self.patterns = self._identify_patterns()
        self.logger.info(f"Identified {len(self.patterns)} patterns in the data")
        
        # Generate automation suggestions
        self.automation_suggestions = self._generate_automation_suggestions()
        self.logger.info(f"Generated {len(self.automation_suggestions)} automation suggestions")
        
        return {
            "patterns": self.patterns,
            "automation_suggestions": self.automation_suggestions
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