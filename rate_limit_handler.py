import time
import random
import functools

class RateLimitHandler:
    """
    A utility class to handle API rate limits with exponential backoff
    and request batching
    """
    
    def __init__(self, max_retries=5, initial_backoff=2, max_backoff=60):
        """
        Initialize the rate limit handler
        
        Args:
            max_retries (int): Maximum number of retries before failing
            initial_backoff (int): Initial backoff time in seconds
            max_backoff (int): Maximum backoff time in seconds
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
    
    def with_retry(self, func):
        """
        Decorator to add retry logic with exponential backoff
        
        Args:
            func: The function to decorate
            
        Returns:
            The decorated function with retry logic
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = self.initial_backoff
            
            while retries <= self.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if it's a rate limit error (429)
                    if "429" in str(e) and retries < self.max_retries:
                        # Calculate backoff with jitter
                        jitter = random.uniform(0.8, 1.2)
                        sleep_time = min(backoff * jitter, self.max_backoff)
                        
                        print(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        
                        # Increase backoff for next attempt
                        backoff *= 2
                        retries += 1
                    else:
                        # Re-raise the exception for other errors
                        raise
            
            # If we've exhausted retries
            raise Exception(f"Failed after {self.max_retries} retries due to rate limiting")
        
        return wrapper

# Example usage in an agent class:
"""
from rate_limit_handler import RateLimitHandler

class YourAgent:
    def __init__(self):
        self.rate_limiter = RateLimitHandler()
        
    @property
    def answer_question(self):
        # Apply rate limit handling to the API call
        @self.rate_limiter.with_retry
        def _answer_question(question, data):
            # Your API call logic here
            response = make_api_call(question, data)
            return response
        
        return _answer_question
"""

# Advanced functionality for request batching
class RequestBatcher:
    """
    A utility class to batch API requests to reduce rate limiting issues
    """
    
    def __init__(self, batch_size=5, batch_delay=1):
        """
        Initialize the request batcher
        
        Args:
            batch_size (int): Maximum number of requests in a batch
            batch_delay (float): Delay between batches in seconds
        """
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.current_batch = 0
    
    def batch_requests(self, func):
        """
        Decorator to batch requests
        
        Args:
            func: The function to decorate
            
        Returns:
            The decorated function with batching logic
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Increment batch counter
            self.current_batch += 1
            
            # If we've reached the batch size, add a delay
            if self.current_batch >= self.batch_size:
                time.sleep(self.batch_delay)
                self.current_batch = 0
            
            return func(*args, **kwargs)
        
        return wrapper

# Combined implementation for your agent classes
def apply_rate_limit_handling(agent_class):
    """
    Class decorator to apply rate limit handling to all API methods
    
    Args:
        agent_class: The agent class to decorate
        
    Returns:
        The decorated class with rate limit handling
    """
    original_init = agent_class.__init__
    
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.rate_limiter = RateLimitHandler()
        self.request_batcher = RequestBatcher()
        
        # Apply decorators to API methods
        for attr_name in dir(self):
            if attr_name.startswith('_') or not callable(getattr(self, attr_name)):
                continue
            
            attr = getattr(self, attr_name)
            if hasattr(attr, '__call__') and any(keyword in attr_name for keyword in ['query', 'answer', 'generate', 'process']):
                decorated = self.rate_limiter.with_retry(self.request_batcher.batch_requests(attr))
                setattr(self, attr_name, decorated)
    
    agent_class.__init__ = __init__
    return agent_class

# Usage:
"""
from rate_limit_handler import apply_rate_limit_handling

@apply_rate_limit_handling
class YourAgent:
    def __init__(self):
        # Original initialization
        pass
        
    def answer_question(self, question, data):
        # Your API call logic here
        response = make_api_call(question, data)
        return response
"""