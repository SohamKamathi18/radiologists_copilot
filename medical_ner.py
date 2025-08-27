
"""
Medical Report Analyzer - Streamlit Application
"""
import os
import fitz  # PyMuPDF
import streamlit as st
import tempfile
import pandas as pd
import re

from config import load_ner_model
from database import store_to_mysql, fetch_all_reports, search_reports, get_entity_statistics, delete_patient

# NER Configuration
NER_CONFIG = {
    'confidence_threshold': 0.5,  # Minimum confidence score
    'max_entities': 20,           # Maximum entities to display
    'min_text_length': 2,         # Minimum entity text length
    'high_confidence_threshold': 0.8  # Threshold for high confidence entities
}

# Load NER model
ner_pipeline = load_ner_model()

def check_database_connection():
    """Check if database connection is available and tables exist."""
    try:
        # Try to fetch reports to test connection
        fetch_all_reports()
        return True
    except Exception as e:
        return False

def show_no_database_message():
    """Show a consistent message when database is not available."""
    st.info("💡 Upload some reports first to initialize the database.")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close() 
        
        if not text.strip():
            st.error("⚠️ No readable text found in PDF. File might be image-based or corrupted.")
            return None
            
        return text
    except Exception as e:
        st.error(f"❌ Failed to extract text from PDF: {str(e)}")
        return None


def extract_patient_details(text):
    """Extract patient details using comprehensive regex patterns."""
    details = {
        "name": "",
        "age": "",
        "gender": ""
    }
    
    # Name extraction patterns
    name_patterns = [
        r'patient\s+name\s*:\s*(?:(?:mr|mrs|ms|dr)\.?\s+)?([A-Z][A-Z\s]+?)(?=\s*(?:study|age|referring|sex|gender|$|\n))',
        r'(?:patient\s+)?name\s*:\s*(?:(?:mr|mrs|ms|dr)\.?\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})(?=\s*(?:\n|$|study|age|dob|sex|gender|mrn|address|phone))',
        r'\b(?:mr|mrs|ms)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bname\s*:\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',
        r'\b(?:name|patient)\s*[:=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'your patient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r're:.*?for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),\s*MRN',
        r'dear\s+dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[:,]',
        r'patient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s+(?:aged?|age|is)',
    ]
    
    # Age extraction patterns
    age_patterns = [
        r'age\s*[:=]\s*(\d{1,3})(?:\s*(?:years?|yrs?|y\.?o\.?))?',
        r'\((\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)\)',
        r'\((\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)\)',
        r'(?:age|aged?)\s*[:=]?\s*(\d{1,3})\s*(?:years?|yrs?|y\.o\.?)?',
        r'(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)',
        r'aged?\s+(\d{1,3})',
        r'age\s*[:=]\s*(\d{1,3})',
        r'DOB:\s*\d{2}/\d{2}/\d{4}\s*\((\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)\)',
        r'(\d{1,3})\s*[-]?\s*year[-\s]*old',
    ]
    
    # Gender extraction patterns
    gender_patterns = [
        r'(?:gender|sex)\s*[:=]\s*(male|female|m|f)',
        r'\b(male|female)\b(?!\s*(?:patient|doctor|nurse))',
        r'(?:mr\.?|male)\b',  # Male indicators
        r'(?:mrs\.?|ms\.?|female)\b',  # Female indicators
    ]
    
    # Extract name
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Get the first non-empty group
            name = next((group for group in match.groups() if group), "").strip()
            if name and len(name) > 1 and len(name) < 50:
                # Clean up the name (remove extra spaces, common prefixes)
                name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
                name = re.sub(r'^(?:mr\.?|mrs\.?|ms\.?|dr\.?)\s*', '', name, flags=re.IGNORECASE)
                if name and not re.match(r'^\d+$', name):  # Not just numbers
                    details["name"] = name.title()
                    break
    
    # Extract age
    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age_str = match.group(1)
            try:
                age = int(age_str)
                if 0 <= age <= 120:  # Reasonable age range
                    details["age"] = str(age)
                    break
            except ValueError:
                continue
    
    # Extract gender
    for pattern in gender_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gender_text = match.group(1) if match.groups() else match.group(0)
            gender_lower = gender_text.lower()
            
            if gender_lower in ['male', 'm', 'mr', 'mr.']:
                details["gender"] = "Male"
                break
            elif gender_lower in ['female', 'f', 'mrs', 'mrs.', 'ms', 'ms.']:
                details["gender"] = "Female"
                break
    
    # Fallback: Search for gender keywords in broader context
    if not details["gender"]:
        if re.search(r'\bmale\b(?!\s*(?:patient|doctor|nurse))', text, re.IGNORECASE):
            # Check if 'female' appears nearby to avoid false positives
            if not re.search(r'\bfemale\b', text, re.IGNORECASE):
                details["gender"] = "Male"
        elif re.search(r'\bfemale\b', text, re.IGNORECASE):
            details["gender"] = "Female"
    
    return details

def extract_ner_entities(text):
    """Process text through NER pipeline with enhanced filtering."""
    try:
        if not text or not text.strip():
            st.warning("⚠️ No text provided for NER processing")
            return []
            
        entities = ner_pipeline(text)
        
        # Enhanced filtering with multiple criteria
        filtered_entities = []
        seen_entities = set()  # To avoid duplicates
        
        for entity in entities:
            confidence = entity.get('score', 0)
            text_content = entity.get('word', '').strip()
            label = entity.get('entity_group', 'UNKNOWN')
            
            # Skip if confidence too low
            if confidence < NER_CONFIG['confidence_threshold']:
                continue
            
            # Skip very short or invalid text
            if len(text_content) < NER_CONFIG['min_text_length']:
                continue
                
            # Skip entities that are just numbers or special characters
            if text_content.isdigit() or not any(c.isalpha() for c in text_content):
                continue
                
            # Skip common non-medical words that get misclassified
            skip_words = {
                'yes', 'no', 'male', 'female', 'work', 'employment', 'report',
                'date', 'name', 'full', 'registration', 'passport', 'residence',
                'worker', 'foreign', 'my', 'fit', 'dr', 'md'
            }
            if text_content.lower() in skip_words:
                continue
            
            # Skip fragmented tokens (those starting with ##)
            if text_content.startswith('##'):
                continue
            
            # Skip single characters
            if len(text_content) == 1 or text_content.lower() in ['a', 'b', 'c', 'x', 'op']:
                continue
            
            # Create a unique key to avoid duplicates
            entity_key = (text_content.lower(), label)
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            
            # Only keep high-confidence medical entities
            medical_labels = {
                'Disease_disorder', 'Medication', 'Diagnostic_procedure',
                'Therapeutic_procedure', 'Biological_structure', 'Sign_symptom'
            }
            
            if label in medical_labels or confidence > NER_CONFIG['high_confidence_threshold']:
                filtered_entities.append({
                    'label': label,
                    'text': text_content,
                    'confidence': round(confidence, 2)
                })
        
        # Sort by confidence (highest first) and limit results
        filtered_entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return top entities to avoid overwhelming display
        return filtered_entities[:NER_CONFIG['max_entities']]
        
    except Exception as e:
        st.error(f"❌ NER processing failed: {str(e)}")
        return []


# Streamlit UI
st.title("Medical Report Analyzer")

# Sidebar configuration
st.sidebar.header("🔧 NER Settings")
NER_CONFIG['confidence_threshold'] = st.sidebar.slider(
    "Minimum Confidence Score", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.1,
    help="Lower values show more entities but with lower confidence"
)
NER_CONFIG['max_entities'] = st.sidebar.slider(
    "Maximum Entities to Show", 
    min_value=5, 
    max_value=50, 
    value=20, 
    step=5
)

menu = st.sidebar.selectbox("Choose an option", ["Upload Report", "View Reports", "Search Reports", "Statistics"])

if menu == "Upload Report":
    st.info("📋 Upload medical reports in PDF format for analysis")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Maximum file size: 10MB per file"
    )

    if uploaded_files:
        # Validate file sizes
        for uploaded_file in uploaded_files:
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.error(f"❌ File {uploaded_file.name} is too large (max 10MB)")
                continue
        
        # Process files with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"Processing {uploaded_file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.subheader(f"📄 {uploaded_file.name}")
            
            text = extract_text_from_pdf(tmp_file_path)
            if not text:
                continue  # Skip if text extraction failed
                
            patient_details = extract_patient_details(text)

            # Show results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**👤 Patient Details:**")
                if any(patient_details.values()):
                    st.json(patient_details)
                else:
                    st.warning("⚠️ No patient details found")

            with col2:
                st.write("**🏥 Medical Entities:**")
                ner_results = extract_ner_entities(text)
                
                if ner_results:
                    for ent in ner_results:
                        confidence_color = "🟢" if ent['confidence'] > 0.8 else "🟡" if ent['confidence'] > 0.5 else "🔴"
                        st.markdown(f"{confidence_color} **{ent['label']}**: {ent['text']} `({ent['confidence']})`")
                else:
                    st.warning("⚠️ No medical entities detected")

            # Only store if we have valid data
            if text and (any(patient_details.values()) or ner_results):
                store_to_mysql(patient_details, ner_results, uploaded_file.name)
                st.success(f"✅ {uploaded_file.name} processed successfully")
            else:
                st.warning(f"⚠️ {uploaded_file.name} contained no useful data")

            os.remove(tmp_file_path)
            st.divider()

        status_text.text("✅ All files processed!")

elif menu == "View Reports":
    st.subheader("📊 Stored Medical Reports")
    
    if not check_database_connection():
        show_no_database_message()
    else:
        try:
            reports = fetch_all_reports()
            
            if not reports:
                st.info("📋 No reports found. Upload some PDF files first!")
            else:
                # Add search filter
                search_filter = st.text_input("🔍 Filter by patient name or ID:")
                for patient in reports:
                    # Apply filter
                    if search_filter and search_filter.lower() not in patient.get('name', '').lower():
                        continue
                        
                    with st.expander(f"👤 Patient: {patient.get('name', 'Unknown')} (ID: {patient['id']})"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Age", patient.get('age', 'N/A'))
                        with col2:
                            st.metric("Gender", patient.get('gender', 'N/A'))
                        with col3:
                            st.metric("Reports", len(patient.get('reports', [])))
                        with col4:
                            # Delete patient button
                            if st.button(f"🗑️ Delete", key=f"delete_{patient['id']}", type="secondary"):
                                # Show confirmation dialog
                                st.session_state[f"confirm_delete_{patient['id']}"] = True
                        
                        # Confirmation dialog
                        if st.session_state.get(f"confirm_delete_{patient['id']}", False):
                            st.warning(f"⚠️ Are you sure you want to delete patient '{patient.get('name', 'Unknown')}'? This will permanently remove all their reports and medical data.")
                            
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅ Yes, Delete", key=f"confirm_yes_{patient['id']}", type="primary"):
                                    result = delete_patient(patient['id'])
                                    if result['success']:
                                        st.success(result['message'])
                                        st.session_state[f"confirm_delete_{patient['id']}"] = False
                                        st.rerun()  # Refresh the page to update the list
                                    else:
                                        st.error(result['message'])
                            
                            with col_no:
                                if st.button("❌ Cancel", key=f"confirm_no_{patient['id']}", type="secondary"):
                                    st.session_state[f"confirm_delete_{patient['id']}"] = False
                                    st.rerun()
                        
                        if patient.get('reports'):
                            for report in patient['reports']:
                                st.write(f"📄 **{report['filename']}** (Processed: {report['processed']})")
                                
                                if report.get('entities'):
                                    entity_df = pd.DataFrame(report['entities'])
                                    st.dataframe(entity_df, use_container_width=True)
                                else:
                                    st.info("No entities found in this report")
                                st.divider()
                        else:
                            st.info("No reports for this patient")
                            
        except Exception as e:
            st.error(f"❌ Failed to load reports: {str(e)}")
            show_no_database_message()

elif menu == "Search Reports":
    st.subheader("🔍 Search Medical Reports")
    
    if not check_database_connection():
        show_no_database_message()
    else:
        query = st.text_input("Enter patient name, ID, or medical entity")
        if query:
            try:
                results = search_reports(query)
                if results:
                    for result in results:
                        st.markdown(f"### Patient ID: {result['id']} | Name: {result['name']}")
                        st.write(f"**Age**: {result['age']}, **Gender**: {result['gender']}")
                else:
                    st.warning("No matching reports found.")
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                show_no_database_message()

elif menu == "Statistics":
    st.subheader("📈 Analytics Dashboard")
    
    if not check_database_connection():
        show_no_database_message()
    else:
        try:
            stats = get_entity_statistics()
            
            if stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🏷️ Entity Frequency**")
                    df = pd.DataFrame.from_dict(stats, orient='index', columns=['Count'])
                    st.bar_chart(df)
                
                with col2:
                    st.write("**📊 Top Entities**")
                    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10])
                    for entity, count in sorted_stats.items():
                        st.metric(entity.replace('_', ' ').title(), count)
                
                # Summary metrics
                st.divider()
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric("Total Entities", sum(stats.values()))
                with col4:
                    st.metric("Unique Entity Types", len(stats))
                with col5:
                    most_common = max(stats.items(), key=lambda x: x[1])
                    st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")
                    
            else:
                st.info("📋 No statistics available. Process some reports first!")
                
        except Exception as e:
            st.error(f"❌ Failed to load statistics: {str(e)}")
            show_no_database_message()
