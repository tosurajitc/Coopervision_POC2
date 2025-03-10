#!/usr/bin/env python
"""
This script downloads the required NLTK data for the Ticket Automation Advisor.
Run this script before starting the application if you encounter NLTK data errors.
"""

import nltk
import os
import sys

def download_nltk_data():
    """Download required NLTK data packages."""
    print("Downloading NLTK data...")
    
    # Create data directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download required packages
    packages = ['punkt', 'stopwords', 'wordnet']
    
    for package in packages:
        print(f"Downloading {package}...")
        try:
            nltk.download(package, download_dir=nltk_data_dir, quiet=False)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
    
    print("\nNLTK data download completed.")
    print(f"Data directory: {nltk_data_dir}")
    print("You can now run the Ticket Automation Advisor.")

if __name__ == "__main__":
    download_nltk_data()