"""
Configuration and model loading for Medical NER application.
"""
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "d4data/biomedical-ner-all"

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', ''),
    'database': os.environ.get('MYSQL_DATABASE', 'medical_ner')
}

# Load NER model
@st.cache_resource
def load_ner_model():
    """Load and cache the NER model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")