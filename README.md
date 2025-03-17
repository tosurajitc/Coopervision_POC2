# Coopervision AI-Driven Automation Analysis

This application analyzes ticket data to identify automation opportunities and provide qualitative insights for IT support teams.

## Features

- **Data Processing**: Automatically process uploaded ticket data files (.csv or .xlsx)
- **Automation Identification**: Identify potential automation opportunities based on ticket patterns
- **Qualitative Insights**: Get concise, actionable insights rather than just quantitative reports
- **Implementation Plans**: Generate step-by-step implementation plans for automation opportunities
- **Custom Queries**: Ask ad-hoc questions about your ticket data

## System Architecture

The application is built with four AI agent components:

1. **Data Processing Agent**: Cleanses and structures uploaded ticket data
2. **Insight Generation Agent**: Analyzes patterns and identifies automation opportunities
3. **Implementation Strategy Agent**: Creates implementation plans for automation suggestions
4. **User Query Agent**: Responds to custom queries about the ticket data

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- GROQ API key (for optimal performance)

### Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on the `.env.template` file:
   ```
   cp .env.template .env
   ```
5. Add your GROQ API key to the `.env` file

### Running the Application

Start the Streamlit application:
```
streamlit run main.py
```

The application will be accessible at `http://localhost:8501`

## Usage

1. **Upload Data**: Upload your ticket data file (.csv or .xlsx)
2. **Optional**: Upload a keywords file (.txt) for targeted analysis
3. **Explore Insights**:
   - View predefined questions analysis
   - Check the suggested automation opportunities
   - Ask custom queries about your data
4. **Implementation Plans**: Get detailed implementation plans for any identified opportunity

## Data Format

The ticket data file should contain the following minimum fields:
- `ticket_id` (or similar identifier)
- `description` (ticket description/summary)
- `resolution` (how the ticket was resolved)
- `assignment_group` (team or group assigned to the ticket)

Additional fields that enhance analysis:
- `closed_notes` (notes added when closing the ticket)
- `comments` (additional comments or updates)
- `status` (ticket status)
- `priority` (ticket priority)
- `created_date` (when the ticket was created)
- `resolved_date` (when the ticket was resolved)

## License

NA
