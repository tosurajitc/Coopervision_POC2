import os
import re
from .groq_client import create_groq_client, call_groq_api
from rate_limit_handler import apply_rate_limit_handling

@apply_rate_limit_handling
class ImplementationStrategyAgent:
    """
    Agent 3: Implementation Strategy
    - Generates detailed, actionable implementation plans for automation suggestions
    - Plans include tools, timelines, phases, and success metrics
    - Focuses on practical execution steps rather than repeating the opportunity description
    """
    
    def __init__(self):
        # Initialize GROQ client
        self.client, self.model_name, self.error_message = create_groq_client()
        if self.error_message:
            print(f"Warning: {self.error_message}")
            
        # Dictionary of implementation templates for common automation categories
        self.implementation_templates = {
            'password_reset': {
                'tools': ['Identity Management System', 'Self-Service Portal', 'Authentication API'],
                'phases': [
                    'Requirements & Security Planning (2 weeks)',
                    'Portal Development & Integration (4 weeks)',
                    'Security Testing & Validation (2 weeks)',
                    'Pilot Rollout & User Training (2 weeks)'
                ],
                'metrics': [
                    'Reduction in password reset tickets (target: 90%)',
                    'Average time for user to reset password (target: <2 minutes)',
                    'User satisfaction with reset process (target: >4.5/5)'
                ]
            },
            'access_request': {
                'tools': ['Workflow Automation Platform', 'Identity and Access Management (IAM) System', 'Integration APIs'],
                'phases': [
                    'Process Mapping & Approval Workflow Design (3 weeks)',
                    'IAM Integration & Form Development (4 weeks)',
                    'Automated Provisioning Setup (3 weeks)',
                    'Testing & Security Validation (2 weeks)'
                ],
                'metrics': [
                    'Reduction in access request processing time (target: 70%)',
                    'First-time approval rate (target: >90%)',
                    'Compliance with access management policies (target: 100%)'
                ]
            },
            'data_entry': {
                'tools': ['Robotic Process Automation (RPA) Platform', 'Data Validation Tools', 'Integration APIs'],
                'phases': [
                    'Process Documentation & Analysis (2 weeks)',
                    'RPA Bot Development (4 weeks)',
                    'Error Handling & Exception Process Creation (2 weeks)',
                    'Parallel Processing & Validation (3 weeks)'
                ],
                'metrics': [
                    'Reduction in manual data entry hours (target: >80%)',
                    'Data accuracy improvement (target: >99% accuracy)',
                    'Processing time reduction (target: 60%)'
                ]
            },
            'report_generation': {
                'tools': ['Business Intelligence Platform', 'Automated Scheduling System', 'Data Warehouse'],
                'phases': [
                    'Report Requirements & Data Source Mapping (3 weeks)',
                    'Dashboard & Template Development (4 weeks)',
                    'Automated Delivery System Setup (2 weeks)',
                    'User Training & Feedback Collection (2 weeks)'
                ],
                'metrics': [
                    'Reduction in manual report creation time (target: 90%)',
                    'Report accuracy rate (target: 100%)',
                    'User adoption of self-service reporting (target: >70%)'
                ]
            }
        }
    
    def generate_plan(self, question, answer):
        """
        Generate an implementation plan based on a question and its answer
        
        Args:
            question (str): The original question
            answer (str): The answer/insight provided
            
        Returns:
            str: A detailed implementation plan
        """
        if not self.client:
            return self._get_fallback_plan(question, answer)
        
        try:
            # Extract key information from the question and answer
            category = self._detect_automation_category(question + " " + answer)
            
            # Prepare a more specific prompt for the LLM
            prompt = f"""
            Create a detailed implementation plan for a technical automation project based on the following information.
            
            Question: "{question}"
            Analysis: "{answer}"
            
            Your implementation plan should be substantially MORE DETAILED than the analysis and focus on HOW to implement the solution, not WHAT the solution is.
            
            Include the following sections with SPECIFIC, ACTIONABLE details:

            1. An objective statement for the implementation project
            2. Specific tools, technologies, and resources required (be specific with actual tool names, not generic categories)
            3. A phased implementation approach with realistic timeframes for each phase
            4. Specific roles and responsibilities for the implementation team
            5. Detailed technical integration points and considerations
            6. Concrete success metrics with quantitative targets
            7. Potential challenges and mitigation strategies
            
            FORMAT YOUR RESPONSE USING THE FOLLOWING TEMPLATE:

# Implementation Plan

## Objective
[Write a specific, measurable objective for the implementation]

## Required Tools & Technologies
- [Specific tool/technology 1]: [Brief explanation of how it will be used]
- [Specific tool/technology 2]: [Brief explanation of how it will be used]
- [Specific tool/technology 3]: [Brief explanation of how it will be used]

## Team Composition
- [Role 1]: [Responsibilities]
- [Role 2]: [Responsibilities]
- [Role 3]: [Responsibilities]

## Implementation Phases
1. **[Phase 1 Name] (Weeks X-Y)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

2. **[Phase 2 Name] (Weeks Z-W)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

3. **[Phase 3 Name] (Weeks J-K)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

4. **[Phase 4 Name] (Weeks L-M)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

## Technical Integration Points
- [Integration point 1]: [Technical details]
- [Integration point 2]: [Technical details]
- [Integration point 3]: [Technical details]

## Success Metrics
- [Specific metric 1]: [Target value]
- [Specific metric 2]: [Target value]
- [Specific metric 3]: [Target value]

## Risk Mitigation
- [Risk 1]: [Mitigation strategy]
- [Risk 2]: [Mitigation strategy]
- [Risk 3]: [Mitigation strategy]

Make sure your implementation plan is highly technical, specific, and actionable - NOT generic. Focus on HOW to implement, not WHAT to implement.
            """
            
            # If we have a template for this category, enhance the prompt
            if category in self.implementation_templates:
                template = self.implementation_templates[category]
                prompt += f"\n\nConsider including these specific tools: {', '.join(template['tools'])}"
                prompt += f"\n\nSuggested phases might include: {'; '.join(template['phases'])}"
                prompt += f"\n\nExample success metrics: {'; '.join(template['metrics'])}"
            
            # Call the GROQ API with increased token allowance for more detailed responses
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=1500,  # Increased for more detailed plans
                temperature=0.2
            )
            
            if error:
                print(f"Error generating implementation plan: {error}")
                return self._get_fallback_plan(question, answer)
                
            # Clean up the response and ensure it has the right format
            response_text = self._clean_and_format_response(response_text, category)
            
            return response_text
            
        except Exception as e:
            print(f"Error generating implementation plan: {str(e)}")
            return self._get_fallback_plan(question, answer)
    
    def generate_plan_for_opportunity(self, opportunity):
        """
        Generate an implementation plan for a specific automation opportunity
        
        Args:
            opportunity (dict): The automation opportunity with title, description, and other fields
            
        Returns:
            str: A detailed implementation plan
        """
        # Extract all available fields for better context
        title = opportunity.get('title', '')
        description = opportunity.get('description', '')
        issue = opportunity.get('issue', '')
        solution = opportunity.get('solution', '')
        justification = opportunity.get('justification', '')
        impact = opportunity.get('impact', 'medium')
        category = opportunity.get('category', '')
        
        # Combine all information for better context
        combined_info = f"Title: {title}\n\n"
        if issue:
            combined_info += f"Issue: {issue}\n\n"
        if solution:
            combined_info += f"Solution: {solution}\n\n"
        if description and description not in combined_info:
            combined_info += f"Description: {description}\n\n"
        if justification:
            combined_info += f"Justification: {justification}\n\n"
        if impact:
            combined_info += f"Impact: {impact}\n\n"
        
        if not self.client:
            return self._get_fallback_plan_for_opportunity(title, combined_info)
        
        try:
            # Detect the automation category for better templating
            detected_category = category if category in self.implementation_templates else self._detect_automation_category(combined_info)
            
            # Prepare an enhanced prompt for the LLM
            prompt = f"""
            Create a detailed technical implementation plan for the following automation opportunity:
            
            {combined_info}
            
            Your implementation plan should be substantially MORE DETAILED than the opportunity description and focus on HOW to implement the solution, not WHAT the solution is.
            
            Include the following sections with SPECIFIC, ACTIONABLE details:

            1. An objective statement for the implementation project
            2. Specific tools, technologies, and resources required (be specific with actual tool names, not generic categories)
            3. A phased implementation approach with realistic timeframes for each phase 
            4. Specific roles and responsibilities for the implementation team
            5. Detailed technical integration points and considerations
            6. Concrete success metrics with quantitative targets
            7. Potential challenges and mitigation strategies
            
            FORMAT YOUR RESPONSE USING THE FOLLOWING TEMPLATE:

# Implementation Plan

## Objective
[Write a specific, measurable objective for the implementation]

## Required Tools & Technologies
- [Specific tool/technology 1]: [Brief explanation of how it will be used]
- [Specific tool/technology 2]: [Brief explanation of how it will be used]
- [Specific tool/technology 3]: [Brief explanation of how it will be used]

## Team Composition
- [Role 1]: [Responsibilities]
- [Role 2]: [Responsibilities]
- [Role 3]: [Responsibilities]

## Implementation Phases
1. **[Phase 1 Name] (Weeks X-Y)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

2. **[Phase 2 Name] (Weeks Z-W)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

3. **[Phase 3 Name] (Weeks J-K)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

4. **[Phase 4 Name] (Weeks L-M)**
   - [Specific task 1]
   - [Specific task 2]
   - [Specific task 3]
   - Deliverable: [Concrete output]

## Technical Integration Points
- [Integration point 1]: [Technical details]
- [Integration point 2]: [Technical details]
- [Integration point 3]: [Technical details]

## Success Metrics
- [Specific metric 1]: [Target value]
- [Specific metric 2]: [Target value]
- [Specific metric 3]: [Target value]

## Risk Mitigation
- [Risk 1]: [Mitigation strategy]
- [Risk 2]: [Mitigation strategy]
- [Risk 3]: [Mitigation strategy]

Make sure your implementation plan is highly technical, specific, and actionable - NOT generic. Focus on HOW to implement, not WHAT to implement.
            """
            
            # If we have a template for this category, enhance the prompt
            if detected_category in self.implementation_templates:
                template = self.implementation_templates[detected_category]
                prompt += f"\n\nConsider including these specific tools: {', '.join(template['tools'])}"
                prompt += f"\n\nSuggested phases might include: {'; '.join(template['phases'])}"
                prompt += f"\n\nExample success metrics: {'; '.join(template['metrics'])}"
            
            # Call the GROQ API with increased token allowance
            response_text, error = call_groq_api(
                self.client, 
                self.model_name,
                prompt,
                max_tokens=1500,  # Increased for more detailed plans
                temperature=0.2
            )
            
            if error:
                print(f"Error generating implementation plan: {error}")
                return self._get_fallback_plan_for_opportunity(title, combined_info)
                
            # Clean up the response and ensure proper formatting
            response_text = self._clean_and_format_response(response_text, detected_category)
            
            return response_text
            
        except Exception as e:
            print(f"Error generating implementation plan: {str(e)}")
            return self._get_fallback_plan_for_opportunity(title, combined_info)
    
    def _detect_automation_category(self, text):
        """
        Detect the automation category from text to use appropriate templates
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: The detected category or None if no match
        """
        text = text.lower()
        
        # Define category keywords
        category_keywords = {
            'password_reset': ['password', 'reset', 'login', 'credentials', 'forgot password'],
            'access_request': ['access', 'permission', 'authorize', 'account access', 'grant access'],
            'data_entry': ['data entry', 'manual entry', 'input data', 'form', 'spreadsheet', 'excel'],
            'report_generation': ['report', 'dashboard', 'analytics', 'export', 'data export', 'visualization'],
            'account_management': ['account', 'user account', 'onboarding', 'offboarding', 'create account'],
            'system_alerts': ['alert', 'notification', 'monitoring', 'system health', 'outage'],
            'routine_maintenance': ['maintenance', 'update', 'patch', 'backup', 'cleanup', 'scheduled'],
            'self_service': ['self-service', 'self service', 'knowledge base', 'faq', 'documentation'],
            'workflow_automation': ['workflow', 'approval', 'routing', 'assignment', 'escalation']
        }
        
        # Count matches for each category
        matches = {category: 0 for category in category_keywords}
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    matches[category] += 1
        
        # Get the category with the most matches
        if matches:
            best_match = max(matches.items(), key=lambda x: x[1])
            if best_match[1] > 0:
                return best_match[0]
        
        # Default category if no matches
        return 'workflow_automation'
    
    def _clean_and_format_response(self, text, category=None):
        """
        Clean the response and ensure it has all required sections
        
        Args:
            text (str): The raw response text
            category (str, optional): The automation category for fallback templates
            
        Returns:
            str: The cleaned and formatted response
        """
        if not text:
            return self._get_generic_implementation_plan(category)
        
        # Remove HTML and thinking content
        text = re.sub(r'<[^>]+>', '', text)
        
        # Check if the response starts with the implementation plan header
        if not text.strip().startswith("# Implementation Plan"):
            # Look for the implementation plan section
            plan_match = re.search(r'# Implementation Plan', text)
            if plan_match:
                # Keep only the implementation plan part
                text = text[plan_match.start():]
            else:
                # Add the header if missing
                text = "# Implementation Plan\n\n" + text
        
        # Check for required sections and add them if missing
        required_sections = [
            "## Objective",
            "## Required Tools & Technologies",
            "## Team Composition",
            "## Implementation Phases",
            "## Success Metrics"
        ]
        
        # Check if sections are present
        missing_sections = []
        for section in required_sections:
            if section not in text:
                missing_sections.append(section)
        
        # If important sections are missing, use the template
        if missing_sections and missing_sections[0] != "## Team Composition":  # Allow missing team composition
            return self._get_generic_implementation_plan(category)
        
        # Add any missing sections from templates
        for section in missing_sections:
            if section == "## Team Composition":
                # Add team composition section if missing
                if "## Implementation Phases" in text:
                    # Add before Implementation Phases
                    text = text.replace("## Implementation Phases", "## Team Composition\n- Project Manager: Overall coordination and stakeholder management\n- Technical Lead: Design and implementation oversight\n- Developers: Building and testing the solution\n- Change Manager: Training and user adoption\n\n## Implementation Phases")
                else:
                    # Add at the end
                    text += "\n\n## Team Composition\n- Project Manager: Overall coordination and stakeholder management\n- Technical Lead: Design and implementation oversight\n- Developers: Building and testing the solution\n- Change Manager: Training and user adoption"
        
        return text
    
    def _get_fallback_plan(self, question, answer):
        """
        Generate a fallback implementation plan when the LLM is not available
        
        Args:
            question (str): The original question
            answer (str): The answer/insight provided
            
        Returns:
            str: A detailed implementation plan
        """
        # Detect the automation category
        category = self._detect_automation_category(question + " " + answer)
        return self._get_generic_implementation_plan(category)
    
    def _get_fallback_plan_for_opportunity(self, title, description):
        """
        Generate a fallback implementation plan for an opportunity when the LLM is not available
        
        Args:
            title (str): The opportunity title
            description (str): The opportunity description
            
        Returns:
            str: A detailed implementation plan
        """
        # Detect the automation category
        category = self._detect_automation_category(title + " " + description)
        return self._get_generic_implementation_plan(category)
    
    def _get_generic_implementation_plan(self, category=None):
        """
        Generate a generic implementation plan, potentially based on a category
        
        Args:
            category (str, optional): The automation category
            
        Returns:
            str: A detailed implementation plan
        """
        # If we have a template for this category, use it
        if category and category in self.implementation_templates:
            template = self.implementation_templates[category]
            tools = "\n".join([f"- {tool}" for tool in template['tools']])
            phases = "\n".join([f"{i+1}. **Phase {i+1}: {phase}**" for i, phase in enumerate(template['phases'])])
            metrics = "\n".join([f"- {metric}" for metric in template['metrics']])
            
            return f"""# Implementation Plan

## Objective
Implement an automated solution for {category.replace('_', ' ')} to improve efficiency, reduce manual effort, and enhance user experience.

## Required Tools & Technologies
{tools}

## Team Composition
- Project Manager: Overall coordination and stakeholder management
- Technical Lead: Design and implementation oversight
- Developers: Building and testing the solution
- Change Manager: Training and user adoption

## Implementation Phases
{phases}

## Technical Integration Points
- User Authentication System: Secure access and identity verification
- Ticketing System API: Integration for automated ticket handling
- Email Notification System: User alerts and status updates

## Success Metrics
{metrics}

## Risk Mitigation
- User Adoption Risk: Comprehensive training and clear communication
- Technical Integration Challenges: Early proof-of-concept testing
- Performance Issues: Load testing and scalability planning
"""
        
        # Otherwise, use a completely generic template
        return """# Implementation Plan

## Objective
Implement an automated solution to reduce manual effort, improve efficiency, and enhance user experience.

## Required Tools & Technologies
- Workflow Automation Platform: For process orchestration and automation
- Integration Middleware: For connecting with existing systems
- Self-Service Portal: For user interaction and request initiation

## Team Composition
- Project Manager: Overall coordination and stakeholder management
- Technical Lead: Design and implementation oversight
- Developers: Building and testing the solution
- Change Manager: Training and user adoption

## Implementation Phases
1. **Analysis & Design (Weeks 1-3)**
   - Document current process in detail
   - Identify integration requirements
   - Design the automation workflow
   - Deliverable: Detailed design document

2. **Development (Weeks 4-7)**
   - Configure automation platform
   - Develop integrations with existing systems
   - Create user interfaces
   - Deliverable: Working prototype

3. **Testing & Refinement (Weeks 8-10)**
   - Conduct user acceptance testing
   - Implement feedback and refinements
   - Perform load and security testing
   - Deliverable: Validated solution

4. **Deployment & Training (Weeks 11-13)**
   - Deploy to production environment
   - Conduct user training sessions
   - Create documentation and knowledge base
   - Deliverable: Fully operational solution

## Technical Integration Points
- User Authentication System: Secure access and identity verification
- Ticketing System API: Integration for automated ticket handling
- Email Notification System: User alerts and status updates

## Success Metrics
- Process Efficiency: 70% reduction in manual processing time
- Error Reduction: 90% decrease in process errors
- User Satisfaction: 85% positive feedback from users

## Risk Mitigation
- User Adoption Risk: Comprehensive training and clear communication
- Technical Integration Challenges: Early proof-of-concept testing
- Performance Issues: Load testing and scalability planning
"""