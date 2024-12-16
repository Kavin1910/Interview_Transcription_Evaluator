import streamlit as st
import docx
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class InterviewAnalyzer:
    def __init__(self):
        """
        Initialize the Interview Transcript Analyzer
        """
        self.evaluation_criteria = {
            'Technical Depth': {
                'keywords': [
                    'salesforce', 'aws', 'cloud', 'crm', 'testing', 
                    'integration', 'technical', 'jira', 'ticket', 'environment'
                ],
                'weight': 0.3
            },
            'Communication Skills': {
                'keywords': [
                    'explain', 'describe', 'clear', 'detailed', 
                    'articulate', 'understand', 'communicate'
                ],
                'weight': 0.2
            },
            'Leadership Potential': {
                'keywords': [
                    'lead', 'team', 'manage', 'report', 'responsible', 
                    'coordinate', 'guide', 'supervisor'
                ],
                'weight': 0.2
            },
            'Experience Depth': {
                'keywords': [
                    'years', 'project', 'work', 'experience', 'domain', 
                    'industry', 'role', 'responsibility'
                ],
                'weight': 0.15
            },
            'Problem-Solving': {
                'keywords': [
                    'solve', 'resolve', 'handle', 'approach', 'challenge', 
                    'issue', 'debug', 'troubleshoot'
                ],
                'weight': 0.15
            }
        }
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Input text
        
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove timestamps and special characters
        text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text
    
    def analyze_transcript(self, text):
        """
        Analyze interview transcript
        
        Args:
            text (str): Full interview transcript
        
        Returns:
            dict: Detailed analysis results
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Initialize scores
        category_scores = {}
        total_weighted_score = 0
        
        # Analyze each category
        for category, details in self.evaluation_criteria.items():
            # Count keyword matches
            keyword_matches = sum(
                1 for keyword in details['keywords'] 
                if keyword in cleaned_text
            )
            
            # Calculate score (max 10 points per category)
            category_score = min(
                10, 
                (keyword_matches / len(details['keywords'])) * 10
            )
            
            # Apply weight
            weighted_score = category_score * details['weight']
            
            category_scores[category] = {
                'raw_score': round(category_score, 2),
                'weighted_score': round(weighted_score, 2)
            }
            
            total_weighted_score += weighted_score
        
        # Calculate overall score
        overall_score = round(total_weighted_score * 10, 2)
        
        # Determine recommendation
        if overall_score >= 85:
            recommendation = {
                'status': 'Highly Recommended',
                'color': 'green',
                'description': 'Exceptional candidate with outstanding potential and comprehensive skills'
            }
        elif overall_score >= 70:
            recommendation = {
                'status': 'Recommended',
                'color': 'blue',
                'description': 'Strong candidate with solid technical and leadership capabilities'
            }
        elif overall_score >= 55:
            recommendation = {
                'status': 'Consider',
                'color': 'orange',
                'description': 'Potential candidate with some promising attributes requiring further evaluation'
            }
        else:
            recommendation = {
                'status': 'Not Recommended',
                'color': 'red',
                'description': 'Candidate does not meet minimum requirements for the role'
            }
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'recommendation': recommendation
        }

def read_docx(uploaded_file):
    """
    Read content from a DOCX file
    
    Args:
        uploaded_file: Uploaded file object
    
    Returns:
        str: Extracted text from the document
    """
    doc = docx.Document(uploaded_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Interview Transcript Analyzer", 
        page_icon="📝", 
        layout="wide"
    )
    
    # Title and description
    st.title("🤝 Interview Transcript Analyzer")
    st.write("Upload an interview transcript for comprehensive analysis")
    
    # Sidebar for additional information
    st.sidebar.header("Interview Details")
    candidate_name = st.sidebar.text_input("Candidate Name", "")
    position = st.sidebar.text_input("Position Applied", "")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Interview Transcript", 
        type=['docx', 'txt'],
        help="Please upload a DOCX or TXT file containing the full interview transcript"
    )
    
    # Initialize analyzer
    analyzer = InterviewAnalyzer()
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.docx'):
                transcript = read_docx(uploaded_file)
            else:
                transcript = uploaded_file.read().decode('utf-8')
            
            # Analyze transcript
            analysis_results = analyzer.analyze_transcript(transcript)
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Overall score and recommendation
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Overall Interview Score", 
                    value=f"{analysis_results['overall_score']}/100"
                )
            
            with col2:
                st.metric(
                    label="Recommendation", 
                    value=analysis_results['recommendation']['status'],
                    help=analysis_results['recommendation']['description']
                )
            
            # Detailed category scores
            st.subheader("Category Breakdown")
            
            # Prepare DataFrame for category scores
            category_df = pd.DataFrame.from_dict(
                analysis_results['category_scores'], 
                orient='index'
            )
            
            # Display category scores
            st.dataframe(category_df, use_container_width=True)
            
            # Recommendation description
            st.info(analysis_results['recommendation']['description'])
            
            # Transcript preview
            with st.expander("View Transcript"):
                st.text(transcript)
        
        except Exception as e:
            st.error(f"Error processing transcript: {str(e)}")

if __name__ == "__main__":
    main()
