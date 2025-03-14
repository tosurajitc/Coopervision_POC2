import pandas as pd
import os
from collections import Counter
import re
import logging
import traceback
import json
from typing import List, Dict, Optional

from .groq_client import create_groq_client, call_groq_api
from rate_limit_handler import apply_rate_limit_handling

@apply_rate_limit_handling
class InsightGenerationAgent:
    """
    Agent 2: Insight Generation & Automation Identification
    - Analyzes ticket data to detect patterns
    - Identifies potential automation opportunities
    - Cross-checks data against keywords if provided
    """
    
    def __init__(
        self, 
        pattern_threshold: float = 0.05,  # 5% of tickets
        reassignment_threshold: float = 0.1,  # 10% of tickets
        long_ticket_threshold: int = 24  # 24 hours
    ):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Configurable thresholds
        self.pattern_threshold = pattern_threshold
        self.reassignment_threshold = reassignment_threshold
        self.long_ticket_threshold = long_ticket_threshold

        # Initialize GROQ client
        self.client, self.model_name, self.error_message = create_groq_client()
        if self.error_message:
            self.logger.warning(f"GROQ Client Initialization Warning: {self.error_message}")
        
        # Define automation patterns to look for - expanded and case-insensitive
        self.automation_patterns = {
            'password_reset': [
                'password reset', 'forgot password', 'reset password', 
                'password expired', 'change password', 'password change',
                'temporary password', 'reset user account'
            ],
            'access_request': [
                'access request', 'permission', 'grant access', 'request access',
                'access to', 'authorize', 'authorization', 'privilege',
                'user access', 'system access', 'account access', 'permissions',
                'access rights', 'role-based', 'user account', 'login issues'
            ],
            'data_entry': [
                'data entry', 'input data', 'enter data', 'form', 'manual entry',
                'spreadsheet', 'excel', 'copy paste', 'copy and paste',
                'data input', 'manual data', 'data transfer', 'data migration',
                'import data', 'export data', 'data upload'
            ],
            'report_generation': [
                'report', 'generate report', 'reporting', 'dashboard', 'monthly report',
                'weekly report', 'data export', 'export', 'extract data',
                'financial report', 'sales report', 'metrics', 'analytics',
                'business intelligence', 'visualization', 'data extract', 'query'
            ],
            'account_management': [
                'create account', 'new account', 'account creation', 'onboarding',
                'offboarding', 'disable account', 'remove account', 'delete user',
                'user creation', 'setup account', 'add user', 'update account',
                'modify user', 'user management', 'account setup'
            ],
            'system_alerts': [
                'alert', 'monitoring', 'notification', 'disk space', 'memory',
                'cpu', 'server down', 'outage', 'system down', 'error message',
                'warning', 'alert message', 'system error', 'crash', 'failure',
                'not responding', 'performance issue', 'slow performance'
            ],
            'routine_maintenance': [
                'maintenance', 'update', 'patch', 'backup', 'cleanup',
                'regular', 'scheduled', 'recurring', 'routine', 'upgrade',
                'software update', 'system update', 'regular maintenance',
                'scheduled task', 'automated backup', 'cleanup process'
            ],
            'printer_scanner_issues': [
                'printer', 'printing', 'scanner', 'scan', 'paper jam', 
                'toner', 'cartridge', 'print queue', 'print job', 'cannot print',
                'printer offline', 'printing error', 'print spooler'
            ],
            'self_service_potential': [
                'how to', 'guide', 'instructions', 'steps', 'procedure',
                'reset', 'change', 'update', 'modify', 'troubleshoot',
                'common issue', 'frequent question', 'known issue'
            ],
            'recurring_workarounds': [
                'workaround', 'temporary solution', 'quick fix', 'temporary fix',
                'interim solution', 'manual correction', 'manual fix',
                'manual process', 'manual update', 'manual workaround'
            ]
        }

    def _normalize_column_names(self, data):
        """
        Normalize column names to handle case sensitivity issues
        """
        # Create a mapping of lowercase column names to actual column names
        column_mapping = {col.lower(): col for col in data.columns}
        
        # Create a new DataFrame with standardized column names
        normalized_data = data.copy()
        
        # Map common column variations to standard names
        field_variations = {
            'description': ['description', 'ticket_description', 'issue_description', 'problem_description'],
            'resolution': ['resolution', 'resolutionnotes', 'resolution_notes', 'notes', 'solution'],
            'ticket_id': ['ticket_id', 'ticketid', 'id', 'ticket_number', 'ticket'],
            'category': ['category', 'ticket_category', 'issue_category', 'type'],
            'duration_hours': ['duration_hours', 'resolution_time', 'resolutiontime', 'time_to_resolve'],
            'priority': ['priority', 'ticket_priority', 'importance', 'severity'],
            'status': ['status', 'ticket_status', 'state'],
            'assigned_team': ['assigned_team', 'assignedteam', 'team', 'assigned_department', 'department'],
            'assigned_agent': ['assigned_agent', 'assignedagent', 'agent', 'assigned_to', 'owner']
        }
        
        # Create new standardized columns based on available data
        for standard_field, variations in field_variations.items():
            for variation in variations:
                if variation.lower() in column_mapping:
                    normalized_data[standard_field] = data[column_mapping[variation.lower()]]
                    break
        
        return normalized_data

    def _check_automation_patterns(self, data):
        """Check for predefined automation patterns in the data"""
        # Make sure we have the right column names
        normalized_data = self._normalize_column_names(data)
        
        results = {category: 0 for category in self.automation_patterns.keys()}
        
        # Determine which fields to check
        description_field = next((field for field in ['description', 'Description'] if field in normalized_data.columns), None)
        resolution_field = next((field for field in ['resolution', 'ResolutionNotes'] if field in normalized_data.columns), None)
        
        if not description_field and not resolution_field:
            self.logger.warning("No description or resolution fields found in data")
            return results
        
        # Check in description and resolution fields
        for idx, row in normalized_data.iterrows():
            description = str(row.get(description_field, '')) if description_field else ''
            resolution = str(row.get(resolution_field, '')) if resolution_field else ''
            combined_text = (description + ' ' + resolution).lower()
            
            for category, patterns in self.automation_patterns.items():
                if any(pattern.lower() in combined_text for pattern in patterns):
                    results[category] += 1
        
        return results

    def _create_opportunity_from_pattern(self, category, count, total_tickets, data):
        """Create an automation opportunity from a detected pattern"""
        percentage = (count / total_tickets) * 100
        
        # Define opportunity based on category
        opportunity_templates = {
            'password_reset': {
                'title': 'Implement Automated Password Reset Solution',
                'issue': f'Identified {count} tickets ({percentage:.1f}%) related to password resets.',
                'solution': 'Implement a self-service password reset portal that allows users to securely reset their passwords without IT intervention.',
                'justification': 'Password resets are simple, repetitive tasks that can be safely automated with proper security controls.',
                'impact': 'high' if percentage > 10 else 'medium'
            },
            'access_request': {
                'title': 'Create Automated Access Request Workflow',
                'issue': f'Found {count} tickets ({percentage:.1f}%) involving access requests or permission changes.',
                'solution': 'Develop a structured workflow system for access requests with automated approval routing and provisioning.',
                'justification': 'Access requests often follow predictable patterns and approval workflows that can be standardized and automated.',
                'impact': 'high' if percentage > 10 else 'medium'
            },
            'data_entry': {
                'title': 'Reduce Manual Data Entry with Automation',
                'issue': f'Detected {count} tickets ({percentage:.1f}%) involving manual data entry or data transfer tasks.',
                'solution': 'Implement data integration or RPA (Robotic Process Automation) solutions to automate repetitive data entry tasks.',
                'justification': 'Manual data entry is time-consuming, error-prone, and a prime candidate for automation.',
                'impact': 'medium'
            },
            'report_generation': {
                'title': 'Automate Recurring Report Generation',
                'issue': f'Identified {count} tickets ({percentage:.1f}%) related to report generation or data extraction.',
                'solution': 'Create a self-service reporting platform with scheduled delivery options for common reports.',
                'justification': 'Automated reporting saves time, ensures consistency, and allows staff to focus on analysis rather than data gathering.',
                'impact': 'medium'
            },
            'account_management': {
                'title': 'Streamline User Account Management',
                'issue': f'Found {count} tickets ({percentage:.1f}%) related to account creation, modification, or deactivation.',
                'solution': 'Implement automated user lifecycle management tied to HR systems for seamless onboarding and offboarding.',
                'justification': 'Automating account management increases security, ensures compliance, and reduces administrative overhead.',
                'impact': 'high' if percentage > 8 else 'medium'
            },
            'system_alerts': {
                'title': 'Enhance Proactive System Monitoring',
                'issue': f'Detected {count} tickets ({percentage:.1f}%) related to system alerts and notifications.',
                'solution': 'Implement advanced monitoring with automated remediation for common issues and intelligent alert routing.',
                'justification': 'Proactive monitoring and automated remediation can prevent issues before they impact users.',
                'impact': 'high' if percentage > 5 else 'medium'
            },
            'routine_maintenance': {
                'title': 'Automate Routine System Maintenance Tasks',
                'issue': f'Identified {count} tickets ({percentage:.1f}%) involving routine maintenance activities.',
                'solution': 'Implement scheduled automation scripts for common maintenance tasks like cleanup, updates, and backups.',
                'justification': 'Routine tasks can be scheduled and automated to run during off-hours without human intervention.',
                'impact': 'medium'
            },
            'printer_scanner_issues': {
                'title': 'Streamline Printer Support with Self-Service Solutions',
                'issue': f'Found {count} tickets ({percentage:.1f}%) related to printer or scanner issues.',
                'solution': 'Implement self-service diagnostics and solution portal for common printing issues with automated driver deployment.',
                'justification': 'Many printing issues follow known patterns that can be resolved through guided troubleshooting.',
                'impact': 'medium'
            },
            'self_service_potential': {
                'title': 'Expand Self-Service Knowledge Base',
                'issue': f'Identified {count} tickets ({percentage:.1f}%) that could potentially be resolved through self-service.',
                'solution': 'Develop a comprehensive self-service portal with guided solutions for common issues.',
                'justification': 'Many user questions follow patterns that could be addressed through well-designed self-service resources.',
                'impact': 'high' if percentage > 15 else 'medium'
            },
            'recurring_workarounds': {
                'title': 'Convert Recurring Workarounds into Permanent Solutions',
                'issue': f'Detected {count} tickets ({percentage:.1f}%) involving recurring workarounds or temporary fixes.',
                'solution': 'Identify patterns in workarounds and develop permanent automated solutions for these recurring issues.',
                'justification': 'Recurring workarounds indicate an opportunity to develop permanent solutions that eliminate the need for manual intervention.',
                'impact': 'high' if percentage > 5 else 'medium'
            }
        }
        
        # Get the template for this category or use a generic one if not found
        template = opportunity_templates.get(category, {
            'title': f'Automate {category.replace("_", " ").title()} Processes',
            'issue': f'Found {count} tickets ({percentage:.1f}%) related to {category.replace("_", " ")}.',
            'solution': 'Implement automation for these recurring tasks to reduce manual effort.',
            'justification': 'Recurring patterns indicate potential for efficiency improvements through automation.',
            'impact': 'medium'
        })
        
        # Create the opportunity
        opportunity = {
            'title': template['title'],
            'issue': template['issue'],
            'solution': template['solution'],
            'justification': template['justification'],
            'impact': template['impact'],
            'description': f"{template['issue']}\n\n{template['solution']}\n\nJustification: {template['justification']}",
            'category': category
        }
        
        return opportunity

    def _get_llm_insights(self, data, num_insights=5):
        """
        Use GROQ LLM to get additional insights
        
        Args:
            data (pd.DataFrame): Processed ticket data
            num_insights (int): Number of insights to request
            
        Returns:
            list: List of insights generated by the LLM
        """
        if not self.client:
            print("GROQ client not available, skipping LLM insights")
            return []
        
        try:
            # Normalize column names for consistency
            normalized_data = self._normalize_column_names(data)
            
            # Prepare a representative sample of the data for analysis
            # Include more tickets and get a diverse sample across categories
            sample_size = min(50, len(normalized_data))
            
            # If category exists, sample across different categories
            if 'category' in normalized_data.columns:
                categories = normalized_data['category'].unique()
                samples = []
                for category in categories:
                    category_data = normalized_data[normalized_data['category'] == category]
                    category_sample = min(int(sample_size / len(categories)), len(category_data))
                    if category_sample > 0:
                        samples.append(category_data.sample(category_sample))
                sample_data = pd.concat(samples)
                if len(sample_data) < sample_size:
                    # Add more random samples to reach desired size
                    remaining = sample_size - len(sample_data)
                    samples.append(normalized_data.sample(remaining))
                    sample_data = pd.concat(samples)
            else:
                # Without categories, just take a random sample
                sample_data = normalized_data.sample(sample_size)
            
            # Extract required fields for analysis
            required_fields = ['ticket_id', 'description', 'resolution', 'category', 'priority']
            available_fields = [field for field in required_fields if field in normalized_data.columns]
            
            # Prepare a more structured data representation
            data_rows = []
            for _, row in sample_data.iterrows():
                data_row = {}
                for field in available_fields:
                    data_row[field] = str(row.get(field, ''))
                data_rows.append(data_row)
            
            # Convert to a JSON string for the prompt
            data_json = json.dumps(data_rows, indent=2)
            
            # Prepare improved prompt for the LLM
            prompt = f"""
            You are an IT automation consultant analyzing support ticket data to identify automation opportunities.
            
            Based on the provided ticket data, identify exactly {num_insights} clear opportunities for automation.
            For each opportunity, provide:
            1. Title - A concise, specific name for the automation opportunity
            2. Issue - The specific problem or pain point identified in the ticket data
            3. Solution - The proposed automation solution to address the issue
            4. Justification - Why this solution is effective and worth implementing
            5. Impact - The potential impact of implementing this automation (high, medium, or low)
            
            Focus on identifying patterns in:
            - Repetitive manual tasks that could be automated
            - Common issues that could be addressed with self-service
            - Process inefficiencies that could be streamlined
            - Knowledge gaps that could be filled with better documentation
            
            Ticket data (JSON format):
            {data_json}
            
            Format your response as a JSON array with {num_insights} objects, each with 'title', 'issue', 'solution', 'justification', and 'impact' fields.
            Be specific and actionable, and base your recommendations directly on patterns observed in the ticket data.
            """
            
            # Call the GROQ API
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            if error:
                print(f"Error getting LLM insights: {error}")
                return []
                
            # Parse the response
            return self._parse_llm_insights(response_text)
        
        except Exception as e:
            print(f"Error getting LLM insights: {str(e)}")
            traceback.print_exc()
            return []

    def _parse_llm_insights(self, insights_text):
        """
        Parse the LLM response to extract structured automation opportunities
        
        Args:
            insights_text (str): The raw text response from the LLM
            
        Returns:
            list: List of structured automation opportunities
        """
        opportunities = []
        
        # Try to parse as JSON first
        try:
            # Look for JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', insights_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                insights = json.loads(json_str)
                
                for insight in insights:
                    opportunity = {
                        'title': insight.get('title', 'Automation Opportunity'),
                        'issue': insight.get('issue', ''),
                        'solution': insight.get('solution', ''),
                        'justification': insight.get('justification', ''),
                        'impact': insight.get('impact', 'medium').lower(),
                        'description': f"{insight.get('issue', '')}\n\n{insight.get('solution', '')}\n\nJustification: {insight.get('justification', '')}",
                        'category': 'llm_identified'
                    }
                    opportunities.append(opportunity)
                return opportunities
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing fails, try regex approach as backup
        try:
            # Split the text into sections for each opportunity
            sections = re.split(r'(?:Opportunity|Insight)\s*\d+:|TITLE:', insights_text)[1:]  # Skip the first empty part
            
            for section in sections:
                try:
                    # Extract fields using regex
                    title_match = re.search(r'(?:Title:)?\s*(.+?)(?=Issue:|ISSUE:|Pattern:|PATTERN:|$)', section, re.IGNORECASE | re.DOTALL)
                    issue_match = re.search(r'(?:Issue|ISSUE):\s*(.+?)(?=Solution:|SOLUTION:|Automation Potential:|AUTOMATION_POTENTIAL:|$)', section, re.IGNORECASE | re.DOTALL)
                    solution_match = re.search(r'(?:Solution|SOLUTION):\s*(.+?)(?=Justification:|JUSTIFICATION:|Impact:|IMPACT:|$)', section, re.IGNORECASE | re.DOTALL)
                    justification_match = re.search(r'(?:Justification|JUSTIFICATION):\s*(.+?)(?=Impact:|IMPACT:|$)', section, re.IGNORECASE | re.DOTALL)
                    impact_match = re.search(r'(?:Impact|IMPACT):\s*(.+?)(?=$|\n)', section, re.IGNORECASE | re.DOTALL)
                    
                    # Use extracted fields or defaults
                    title = title_match.group(1).strip() if title_match else "Automation Opportunity"
                    issue = issue_match.group(1).strip() if issue_match else ""
                    solution = solution_match.group(1).strip() if solution_match else ""
                    justification = justification_match.group(1).strip() if justification_match else ""
                    impact = impact_match.group(1).strip().lower() if impact_match else "medium"
                    
                    # Ensure impact is one of the expected values
                    if impact not in ['high', 'medium', 'low']:
                        impact = "medium"
                    
                    # Create the opportunity object
                    opportunity = {
                        'title': title,
                        'issue': issue,
                        'solution': solution,
                        'justification': justification,
                        'impact': impact,
                        'description': f"{issue}\n\n{solution}\n\nJustification: {justification}",
                        'category': 'llm_identified'
                    }
                    
                    opportunities.append(opportunity)
                except Exception as e:
                    print(f"Error parsing LLM insight section: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error parsing LLM insights with regex: {str(e)}")
        
        return opportunities

    def generate_insights(
        self, 
        data: pd.DataFrame, 
        keywords: Optional[List[str]] = None,
        num_insights: int = 5
    ) -> List[Dict[str, str]]:
        """
        Generate insights and identify automation opportunities
        
        Args:
            data (pd.DataFrame): Processed ticket data
            keywords (list, optional): List of keywords to cross-check
            num_insights (int): Number of insights to generate
            
        Returns:
            list: List of automation opportunities with title and description
        """
        # Normalize column names for case-insensitive matching
        normalized_data = self._normalize_column_names(data)
        
        # Get basic statistics
        total_tickets = len(normalized_data)
        
        # Check for empty dataset
        if total_tickets == 0:
            return [{
                'title': 'No Data Available for Analysis',
                'issue': 'The dataset provided is empty or could not be processed.',
                'solution': 'Please provide valid ticket data for analysis.',
                'justification': 'Automation insights require ticket data to identify patterns.',
                'impact': 'low',
                'description': 'No data available for analysis. Please provide valid ticket data.',
                'category': 'error'
            }]
        
        # Log basic data information
        self.logger.info(f"Analyzing {total_tickets} tickets for automation opportunities")
        self.logger.info(f"Available columns: {normalized_data.columns.tolist()}")
        
        # Identify common patterns
        automation_opportunities = []
        
        # 1. Check for predefined automation patterns
        pattern_results = self._check_automation_patterns(normalized_data)
        for category, count in pattern_results.items():
            if count > 0 and (count / total_tickets) > self.pattern_threshold:
                opportunity = self._create_opportunity_from_pattern(category, count, total_tickets, normalized_data)
                automation_opportunities.append(opportunity)
        
        # 2. Look for reassignment patterns
        reassignment_columns = ['reassignment_indicator', 'reassigned', 'transferred', 'escalated']
        reassignment_column = next((col for col in reassignment_columns if col in normalized_data.columns), None)
        
        if reassignment_column:
            reassigned_tickets = normalized_data[normalized_data[reassignment_column] == True]
            if len(reassigned_tickets) > 0 and (len(reassigned_tickets) / total_tickets) > self.reassignment_threshold:
                opportunity = {
                    'title': 'Reduce Ticket Reassignments with Intelligent Routing',
                    'issue': f'Found {len(reassigned_tickets)} tickets ({(len(reassigned_tickets)/total_tickets)*100:.1f}%) that were reassigned to different teams or agents.',
                    'solution': 'Implement intelligent ticket routing based on keywords, historical patterns, and agent expertise.',
                    'justification': 'Reducing reassignments improves first-contact resolution rates and customer satisfaction while decreasing overall resolution time.',
                    'impact': 'medium',
                    'description': f'Found {len(reassigned_tickets)} tickets ({(len(reassigned_tickets)/total_tickets)*100:.1f}%) that were reassigned. Implementing intelligent ticket routing based on keywords and patterns could reduce handling time and improve first-contact resolution rates.',
                    'category': 'workflow'
                }
                automation_opportunities.append(opportunity)
        
        # 3. Look for long-duration tickets
        duration_columns = ['duration_hours', 'ResolutionTime', 'resolution_time', 'time_spent']
        duration_column = next((col for col in duration_columns if col in normalized_data.columns), None)
        
        if duration_column:
            try:
                # Convert to numeric if not already
                normalized_data[duration_column] = pd.to_numeric(normalized_data[duration_column], errors='coerce')
                
                long_tickets = normalized_data[normalized_data[duration_column] > self.long_ticket_threshold]
                if len(long_tickets) > 0 and (len(long_tickets) / total_tickets) > 0.1:
                    opportunity = {
                        'title': 'Accelerate Resolution for Time-Consuming Tickets',
                        'issue': f'Identified {len(long_tickets)} tickets ({(len(long_tickets)/total_tickets)*100:.1f}%) taking more than {self.long_ticket_threshold} hours to resolve.',
                        'solution': 'Implement automated diagnostics, solution recommendations, and self-service options for common time-consuming issues.',
                        'justification': 'Reducing resolution time for lengthy tickets improves user satisfaction and frees up support resources for more complex issues.',
                        'impact': 'high',
                        'description': f'Identified {len(long_tickets)} tickets ({(len(long_tickets)/total_tickets)*100:.1f}%) taking more than {self.long_ticket_threshold} hours to resolve. Implementing self-service solutions or automated workflows for these issues could significantly reduce resolution times.',
                        'category': 'efficiency'
                    }
                    automation_opportunities.append(opportunity)
            except Exception as e:
                self.logger.warning(f"Error analyzing duration: {str(e)}")
        
        # 4. Cross-check with provided keywords if available
        if keywords and len(keywords) > 0:
            keyword_opportunities = self._analyze_with_keywords(normalized_data, keywords)
            automation_opportunities.extend(keyword_opportunities)
        
        # 5. Use GROQ to get additional insights if available
        llm_insights_count = max(1, num_insights - len(automation_opportunities))
        if self.client and llm_insights_count > 0:
            self.logger.info(f"Requesting {llm_insights_count} insights from LLM")
            try:
                ai_opportunities = self._get_llm_insights(normalized_data, llm_insights_count)
                automation_opportunities.extend(ai_opportunities)
                self.logger.info(f"Received {len(ai_opportunities)} insights from LLM")
            except Exception as e:
                self.logger.error(f"Error getting LLM insights: {str(e)}")
                traceback.print_exc()
        
        # Sort opportunities by potential impact
        automation_opportunities = sorted(
            automation_opportunities, 
            key=lambda x: 0 if 'impact' not in x else (
                0 if x['impact'] == 'high' else (
                    1 if x['impact'] == 'medium' else 2
                )
            )
        )
        
        # Ensure we have the requested number of insights
        if len(automation_opportunities) < num_insights:
            self.logger.warning(f"Only generated {len(automation_opportunities)} insights, padding to {num_insights}")
            # Pad with placeholders
            for i in range(len(automation_opportunities), num_insights):
                placeholders = [
                    {
                        'title': 'Potential Knowledge Base Expansion',
                        'issue': 'Analysis suggests users may benefit from expanded self-service documentation.',
                        'solution': 'Develop a comprehensive knowledge base with guided troubleshooting for common issues.',
                        'justification': 'Self-service documentation reduces ticket volume and empowers users to resolve simple issues independently.',
                        'impact': 'medium'
                    },
                    {
                        'title': 'Automated Ticket Categorization',
                        'issue': 'Manual ticket categorization may lead to inconsistencies and routing delays.',
                        'solution': 'Implement AI-based ticket classification to automatically categorize and route incoming tickets.',
                        'justification': 'Automated categorization improves routing accuracy and reduces initial processing time.',
                        'impact': 'medium'
                    },
                    {
                        'title': 'Regular System Health Checks',
                        'issue': 'Proactive monitoring could prevent some system-related issues before they affect users.',
                        'solution': 'Implement automated system health checks with preventive maintenance protocols.',
                        'justification': 'Preventive maintenance reduces unplanned outages and improves system reliability.',
                        'impact': 'medium'
                    }
                ]
                placeholder = placeholders[i % len(placeholders)]
                placeholder['description'] = f"{placeholder['issue']}\n\n{placeholder['solution']}\n\nJustification: {placeholder['justification']}"
                placeholder['category'] = 'placeholder'
                automation_opportunities.append(placeholder)
        
        # Limit to requested number and return
        return automation_opportunities[:num_insights]
    
    def _analyze_with_keywords(self, data: pd.DataFrame, keywords: List[str]) -> List[Dict[str, str]]:
        """
        Improve keyword analysis with more robust matching
        """
        def fuzzy_keyword_match(text: str, keywords: List[str]) -> List[str]:
            """
            Perform fuzzy matching of keywords in text
            """
            matched_keywords = []
            for keyword in keywords:
                # Case-insensitive partial match
                if keyword.lower() in text.lower():
                    matched_keywords.append(keyword)
            return matched_keywords

        opportunities = []
        
        # Create a word frequency counter
        word_freq = Counter()
        
        # Determine which fields to check
        text_fields = []
        for field in ['description', 'resolution', 'Description', 'ResolutionNotes']:
            if field in data.columns:
                text_fields.append(field)
        
        if not text_fields:
            self.logger.warning("No text fields found for keyword analysis")
            return []
        
        # Check for keywords in available text fields
        keyword_matches = {keyword: 0 for keyword in keywords}
        
        for idx, row in data.iterrows():
            combined_text = ' '.join([str(row.get(field, '')) for field in text_fields]).lower()
            
            # Use fuzzy matching
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    keyword_matches[keyword] += 1
        
        # Find keywords with significant occurrences
        significant_keywords = {
            k: v for k, v in keyword_matches.items() 
            if v > len(data) * self.pattern_threshold
        }
        
        if significant_keywords:
            # Group similar keywords
            grouped_keywords = {}
            for kw, count in significant_keywords.items():
                # Simple grouping based on first word
                key_term = kw.split()[0] if ' ' in kw else kw
                if key_term not in grouped_keywords:
                    grouped_keywords[key_term] = []
                grouped_keywords[key_term].append((kw, count))
            
            # Create opportunities for each group
            for key_term, keyword_counts in grouped_keywords.items():
                total_count = sum(count for _, count in keyword_counts)
                keywords_list = ", ".join([kw for kw, _ in keyword_counts])
                
                # Determine impact based on prevalence
                if total_count > len(data) * 0.2:
                    impact = 'high'
                elif total_count > len(data) * 0.1:
                    impact = 'medium'
                else:
                    impact = 'low'
                
                opportunity = {
                    'title': f'Automation Opportunity for "{key_term.title()}" Related Issues',
                    'issue': f'Found {total_count} tickets ({(total_count/len(data))*100:.1f}%) containing keywords: {keywords_list}.',
                    'solution': f'Create automated solutions or self-service options specifically addressing {key_term}-related issues.',
                    'justification': 'These recurring keyword patterns indicate common issues that could benefit from standardized solutions.',
                    'impact': impact,
                    'description': f'Found {total_count} tickets ({(total_count/len(data))*100:.1f}%) containing keywords: {keywords_list}. These could indicate a repetitive process suitable for automation.',
                    'category': 'keyword_identified',
                    'keywords': [kw for kw, _ in keyword_counts]
                }
                opportunities.append(opportunity)
        
        return opportunities