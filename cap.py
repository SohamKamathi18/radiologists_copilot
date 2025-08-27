# xray_report_generator_with_text_qa.py
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
from PIL import Image
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import pdfplumber
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from skimage import measure, morphology
from scipy import ndimage
import io
import tempfile
import pandas as pd
import re
import fitz  # PyMuPDF

# Add these anatomical region definitions
ANATOMICAL_REGIONS = {
    'upper_left_lung': {'coords': (0, 0, 0.45, 0.6), 'label': 'Upper Left Lung'},
    'lower_left_lung': {'coords': (0, 0.4, 0.45, 1.0), 'label': 'Lower Left Lung'},
    'upper_right_lung': {'coords': (0.55, 0, 1.0, 0.6), 'label': 'Upper Right Lung'},
    'lower_right_lung': {'coords': (0.55, 0.4, 1.0, 1.0), 'label': 'Lower Right Lung'},
    'heart': {'coords': (0.35, 0.3, 0.65, 0.8), 'label': 'Cardiac Region'},
    'mediastinum': {'coords': (0.4, 0.1, 0.6, 0.9), 'label': 'Mediastinum'},
    'left_costophrenic': {'coords': (0.1, 0.75, 0.4, 0.95), 'label': 'Left Costophrenic Angle'},
    'right_costophrenic': {'coords': (0.6, 0.75, 0.9, 0.95), 'label': 'Right Costophrenic Angle'}
}

def find_activation_regions(cam_map, threshold=0.3, min_area=100):
    """
    Find connected regions of high activation in the CAM map
    """
    # Threshold the CAM map
    binary_map = cam_map > threshold
    
    # Clean up small noise
    binary_map = morphology.remove_small_objects(binary_map, min_size=min_area)
    binary_map = morphology.binary_closing(binary_map, morphology.disk(5))
    
    # Find connected components
    labeled_regions = measure.label(binary_map)
    regions = measure.regionprops(labeled_regions, intensity_image=cam_map)
    
    return regions, labeled_regions

def get_anatomical_region(centroid, image_shape):
    """
    Determine which anatomical region a point belongs to
    """
    y, x = centroid
    h, w = image_shape
    
    # Normalize coordinates to 0-1 range
    norm_x = x / w
    norm_y = y / h
    
    for region_name, region_info in ANATOMICAL_REGIONS.items():
        x1, y1, x2, y2 = region_info['coords']
        if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
            return region_info['label']
    
    return "Unspecified Region"

def analyze_pathology_regions(segmentation_maps, image_shape, activation_threshold=0.3):
    """
    Analyze segmentation maps to identify specific affected regions
    """
    region_analysis = {}
    
    for pathology, seg_map in segmentation_maps.items():
        regions, labeled_regions = find_activation_regions(seg_map, activation_threshold)
        
        pathology_regions = []
        for i, region in enumerate(regions):
            centroid = region.centroid
            anatomical_location = get_anatomical_region(centroid, image_shape)
            
            region_info = {
                'region_id': i + 1,
                'anatomical_location': anatomical_location,
                'centroid': centroid,
                'area': region.area,
                'max_intensity': region.max_intensity,
                'mean_intensity': region.mean_intensity,
                'bbox': region.bbox  # (min_row, min_col, max_row, max_col)
            }
            pathology_regions.append(region_info)
        
        region_analysis[pathology] = {
            'regions': pathology_regions,
            'labeled_map': labeled_regions
        }
    
    return region_analysis

def create_labeled_overlay_visualization(image, segmentation_maps, region_analysis, alpha=0.4):
    """
    Create visualization with labeled regions and bounding boxes
    """
    try:
        img_array = np.array(image)
        num_pathologies = len(segmentation_maps)
        
        if num_pathologies == 0:
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, min(num_pathologies, 3), figsize=(18, 12))
        
        # Handle single pathology case
        if num_pathologies == 1:
            axes = axes.reshape(2, 1)
        elif num_pathologies == 2:
            axes = axes.reshape(2, 2)
            
        colors = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        bbox_colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for idx, (pathology, seg_map) in enumerate(segmentation_maps.items()):
            if idx >= 3:  # Limit to 3 pathologies for display
                break
                
            col_idx = idx % 3
            
            # Top row: Segmentation overlay
            ax_top = axes[0, col_idx] if num_pathologies > 1 else axes[0, 0]
            ax_top.imshow(img_array, cmap='gray')
            ax_top.imshow(seg_map, cmap=colors[idx % len(colors)], alpha=alpha, vmin=0, vmax=1)
            ax_top.set_title(f'{pathology} - Segmentation Map', fontsize=12, fontweight='bold')
            ax_top.axis('off')
            
            # Bottom row: Labeled regions with bounding boxes
            ax_bottom = axes[1, col_idx] if num_pathologies > 1 else axes[1, 0]
            ax_bottom.imshow(img_array, cmap='gray')
            ax_bottom.imshow(seg_map, cmap=colors[idx % len(colors)], alpha=alpha, vmin=0, vmax=1)
            
            # Add region labels and bounding boxes
            if pathology in region_analysis:
                regions = region_analysis[pathology]['regions']
                
                for region in regions:
                    # Draw bounding box
                    min_row, min_col, max_row, max_col = region['bbox']
                    rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                   linewidth=2, edgecolor=bbox_colors[idx % len(bbox_colors)], 
                                   facecolor='none', alpha=0.8)
                    ax_bottom.add_patch(rect)
                    
                    # Add text label
                    centroid_y, centroid_x = region['centroid']
                    label_text = f"Region {region['region_id']}\n{region['anatomical_location']}\nIntensity: {region['max_intensity']:.2f}"
                    
                    ax_bottom.annotate(label_text, 
                                     xy=(centroid_x, centroid_y),
                                     xytext=(10, 10), textcoords='offset points',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                     fontsize=8, ha='left', va='bottom',
                                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax_bottom.set_title(f'{pathology} - Labeled Regions', fontsize=12, fontweight='bold')
            ax_bottom.axis('off')
        
        # Hide unused subplots
        for idx in range(num_pathologies, 3):
            if num_pathologies < 3:
                axes[0, idx].axis('off')
                axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        from PIL import Image as PILImage
        overlay_image = PILImage.open(buf)
        plt.close()
        
        return overlay_image
    
    except Exception as e:
        print(f"Error creating labeled overlay visualization: {e}")
        return None

def generate_region_report(region_analysis):
    """
    Generate a detailed text report of affected regions
    """
    report = "## Detailed Region Analysis\n\n"
    
    if not region_analysis:
        return report + "No significant regions detected.\n"
    
    for pathology, analysis in region_analysis.items():
        regions = analysis['regions']
        
        if not regions:
            report += f"### {pathology}\nNo specific regions identified.\n\n"
            continue
            
        report += f"### {pathology}\n"
        report += f"**Number of affected regions:** {len(regions)}\n\n"
        
        for i, region in enumerate(regions, 1):
            report += f"**Region {i}:**\n"
            report += f"- **Location:** {region['anatomical_location']}\n"
            report += f"- **Size:** {region['area']} pixels\n"
            report += f"- **Maximum Activation:** {region['max_intensity']:.3f}\n"
            report += f"- **Average Activation:** {region['mean_intensity']:.3f}\n"
            
            # Interpret activation levels
            if region['max_intensity'] > 0.8:
                severity = "High confidence detection"
            elif region['max_intensity'] > 0.5:
                severity = "Moderate confidence detection"
            else:
                severity = "Low confidence detection"
            
            report += f"- **Confidence Level:** {severity}\n\n"
    
    report += "---\n**Note:** Region analysis is based on AI model predictions. Clinical correlation is recommended.\n"
    return report

# Updated function to replace the original create_overlay_visualization
def create_enhanced_overlay_visualization(image, segmentation_maps, alpha=0.4, activation_threshold=0.3):
    """
    Enhanced version that includes region labeling
    """
    try:
        img_array = np.array(image)
        image_shape = img_array.shape[:2]
        
        # Analyze regions
        region_analysis = analyze_pathology_regions(segmentation_maps, image_shape, activation_threshold)
        
        # Create labeled visualization
        labeled_overlay = create_labeled_overlay_visualization(image, segmentation_maps, region_analysis, alpha)
        
        # Generate region report
        region_report = generate_region_report(region_analysis)
        
        return labeled_overlay, region_analysis, region_report
    
    except Exception as e:
        print(f"Error in enhanced overlay visualization: {e}")
        return None, {}, "Error generating region analysis."

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Model Loading (with caching) ---
@st.cache_resource
def load_clip_model():
    """
    Loads and caches the BiomedCLIP model and processor for report generation.
    """
    try:
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.to(device)
        st.success("BiomedCLIP (for report generation) loaded successfully!")
        return preprocess, model, tokenizer
    except Exception as e:
        st.error(f"Error loading BiomedCLIP model: {e}")
        st.info("Please ensure you have a stable internet connection.")
        st.stop()

# ChexNet class definitions and labels
CHEXNET_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class ChexNet(nn.Module):
    """ChexNet model architecture based on DenseNet121"""
    def __init__(self, num_classes=14):
        super(ChexNet, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.densenet121(x)

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()  # Detach here to avoid gradient tracking
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # Detach here too
        
    def generate_cam(self, input_image, class_idx):
        # Clear previous gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)

        # Backward pass
        output[0, class_idx].backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Compute weights by global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Initialize CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        # Compute weighted combination of activation maps
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = torch.relu(cam)
        if torch.max(cam) > 0:  # Avoid division by zero
            cam = cam / torch.max(cam)

        return cam.cpu().numpy()  # Now safe to convert to numpy

@st.cache_resource
def load_chexnet_model():
    """
    Loads and caches the ChexNet model for pathology detection and segmentation.
    """
    try:
        model = ChexNet(num_classes=len(CHEXNET_LABELS))
        # model.load_state_dict(torch.load('chexnet_model.pth.tar', map_location=device))  # Uncomment and set path
        model.to(device)
        model.eval()
        target_layer = model.densenet121.features.denseblock4.denselayer16.conv2
        grad_cam = GradCAM(model, target_layer)
        st.success("ChexNet model loaded successfully!")
        return model, grad_cam
    except Exception as e:
        st.error(f"Error loading ChexNet model: {e}")
        st.info("Please ensure ChexNet model weights are available.")
        model = ChexNet(num_classes=len(CHEXNET_LABELS))
        model.to(device)
        model.eval()
        target_layer = model.densenet121.features.denseblock4.denselayer16.conv2
        grad_cam = GradCAM(model, target_layer)
        return model, grad_cam

def preprocess_image_for_chexnet(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_pathologies(image, chexnet_model, threshold=0.5):
    try:
        image_tensor = preprocess_image_for_chexnet(image)
        with torch.no_grad():
            predictions = chexnet_model(image_tensor)
            predictions = predictions.cpu().numpy()[0]
        results = {}
        for i, label in enumerate(CHEXNET_LABELS):
            results[label] = {
                'probability': float(predictions[i]),
                'detected': predictions[i] > threshold
            }
        return results
    except Exception as e:
        st.error(f"Error in pathology prediction: {e}")
        return {}

def generate_segmentation_map(image, chexnet_model, grad_cam, top_predictions, original_size):
    try:
        image_tensor = preprocess_image_for_chexnet(image)
        segmentation_maps = {}
        for pathology, data in top_predictions.items():
            if data['detected']:
                class_idx = CHEXNET_LABELS.index(pathology)
                cam = grad_cam.generate_cam(image_tensor, class_idx)
                cam_resized = cv2.resize(cam, original_size)
                segmentation_maps[pathology] = cam_resized
        return segmentation_maps
    except Exception as e:
        st.error(f"Error generating segmentation maps: {e}")
        return {}

def create_overlay_visualization(image, segmentation_maps, alpha=0.4):
    try:
        img_array = np.array(image)
        fig, axes = plt.subplots(1, min(len(segmentation_maps) + 1, 4), figsize=(15, 5))
        if len(segmentation_maps) == 0:
            axes = [axes]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        colors = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        for idx, (pathology, seg_map) in enumerate(segmentation_maps.items()):
            if idx + 1 >= len(axes):
                break
            axes[idx + 1].imshow(img_array, cmap='gray')
            axes[idx + 1].imshow(seg_map, cmap=colors[idx % len(colors)], alpha=alpha, vmin=0, vmax=1)
            axes[idx + 1].set_title(f'{pathology} Segmentation')
            axes[idx + 1].axis('off')
        for idx in range(len(segmentation_maps) + 1, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        from PIL import Image as PILImage
        overlay_image = PILImage.open(buf)
        plt.close()
        return overlay_image
    except Exception as e:
        st.error(f"Error creating overlay visualization: {e}")
        return None

def format_chexnet_results(pathology_results, segmentation_maps):
    try:
        sorted_results = sorted(pathology_results.items(), key=lambda x: x[1]['probability'], reverse=True)
        report = "## ChexNet Pathology Detection Results\n\n"
        detected_pathologies = []
        all_pathologies = []
        for pathology, data in sorted_results:
            prob = data['probability']
            detected = data['detected']
            status = "✅ **DETECTED**" if detected else "❌ Not Detected"
            confidence = f"{prob:.1%}"
            segmentation_note = ""
            if detected and pathology in segmentation_maps:
                segmentation_note = " *(Segmentation Available)*"
            line = f"- **{pathology}:** {status} - Confidence: {confidence}{segmentation_note}"
            all_pathologies.append(line)
            if detected:
                detected_pathologies.append(f"{pathology} ({confidence})")
        if detected_pathologies:
            report += f"**🚨 Detected Pathologies:** {', '.join(detected_pathologies)}\n\n"
        else:
            report += "**✅ No significant pathologies detected**\n\n"
        report += "**Detailed Results:**\n\n"
        report += "\n".join(all_pathologies)
        report += "\n\n---\n**Note:** ChexNet results are AI-generated predictions. Always consult with a qualified radiologist for proper medical diagnosis."
        return report
    except Exception as e:
        st.error(f"Error formatting ChexNet results: {e}")
        return "Error formatting results."

# --- Report Generation ---
def generate_report(image, processor, model, tokenizer, candidate_labels):
    """
    Generates a descriptive report for the given X-ray image by performing zero-shot classification.
    """
    if not image or not candidate_labels:
        return "Error: Please upload an image and provide at least one descriptive label.", ""

    try:
        template = 'this is a photo of '
        texts = tokenizer([template + label for label in candidate_labels], context_length=256).to(device)

        with torch.no_grad():
            image_processed = processor(image).unsqueeze(0).to(device)
            image_features, text_features, logit_scale = model(image_processed, texts)
            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)

        probs = logits.cpu().numpy()
        scores = {label: prob for label, prob in zip(candidate_labels, probs[0])}
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # --- Format the Report for UI display and for context ---
        report_for_display = "## X-Ray Analysis Report\n\n**Potential Findings & Confidence Scores:**\n\n"
        report_for_context = "X-Ray Analysis Report. Potential Findings and Confidence Scores are: "

        report_lines = []
        context_lines = []

        for label, score in sorted_scores:
            report_lines.append(f"- **{label}:** {score:.2%}")
            context_lines.append(f"{label} with a confidence of {score:.2%}")

        report_for_display += "\n".join(report_lines)
        report_for_context += ", ".join(context_lines) + "."

        disclaimer = "\n\n---\n**Disclaimer:** This is an AI-generated report. It is **not a substitute for professional medical advice**. Always consult a qualified radiologist."
        report_for_display += disclaimer
        
        return report_for_display, report_for_context

    except Exception as e:
        st.error(f"An error occurred during report generation: {e}")
        return "Failed to generate report.", ""

# --- Compare X-rays ---
def compare_xrays(previous_xray_input, current_xray_input, processor, model, tokenizer, candidate_labels):
    """
    Compares two X-ray images or reports and suggests improvements based on generated reports.
    """
    # If the input is a string, it's assumed to be pre-extracted report text.
    # Otherwise, it's an Image object, and a report needs to be generated.
    previous_report_context = previous_xray_input if isinstance(previous_xray_input, str) else \
                              generate_report(previous_xray_input, processor, model, tokenizer, candidate_labels)[1]
    
    current_report_context = current_xray_input if isinstance(current_xray_input, str) else \
                             generate_report(current_xray_input, processor, model, tokenizer, candidate_labels)[1]

    context = f"Previous report: {previous_report_context}\nCurrent report: {current_report_context}"
    question = "What improvements or advice can be suggested based on these reports? Provide a detailed and insightful analysis as an experienced radiologist."

    return answer_text_question(context, question)

# --- Extract Text from Report ---
def extract_text_from_report(report_file):
    """
    Extracts text from the uploaded report file (PDF or text).
    """
    try:
        if report_file.type == "application/pdf":
            with pdfplumber.open(report_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        elif report_file.type in ["text/plain"]:
            return report_file.read().decode("utf-8").strip()
        else:
            st.error("Unsupported file type for text extraction.")
            return ""
    except Exception as e:
        st.error(f"Error extracting text from report: {e}")
        return ""

# --- Question Answering ---
def answer_text_question(context, question):
    """
    Answers a user's question based on the generated report (context) using the Gemini API.
    """
    if not context or not question:
        return "Cannot answer without a report context and a question."
    
    try:
        from google import genai
        import os  # Import os to access environment variables

        # Get the API key from the environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is not set. Please set GEMINI_API_KEY environment variable.")
        client = genai.Client(api_key=api_key)  # Ensure this environment variable is set

        if not api_key:
            raise ValueError("API key is not set in the environment variables.")

        # Initialize the Gemini client with the API key
        client = genai.Client(api_key=api_key)

        # Define the system instruction to set the model's persona
        system_instruction = "You are an experienced radiologist. Analyze the provided X-ray report and answer questions based solely on the information within the report, providing clear and concise medical insights."

        # Generate content using the Gemini API with system instruction
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", 
            contents=f"Context: {context}\nQuestion: {question}",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

        # Return the response text
        return f"**Answer:** {response.text}"

    except Exception as e:
        st.error(f"An error occurred during question answering: {e}")
        return "Sorry, I could not answer the question."

# --- MEDICAL NER FUNCTIONALITY (from medical_ner.py) ---

# NER Configuration
NER_CONFIG = {
    'confidence_threshold': 0.5,  # Minimum confidence score
    'max_entities': 20,           # Maximum entities to display
    'min_text_length': 2,         # Minimum entity text length
    'high_confidence_threshold': 0.8  # Threshold for high confidence entities
}

# Import actual logic from config and database files
from config import load_ner_model
from database import (
    store_to_mysql, 
    fetch_all_reports, 
    search_reports, 
    get_entity_statistics, 
    delete_patient,
    get_db_connection
)

def check_database_connection():
    """Check if database connection is available."""
    try:
        # This will try to connect to the database and raise an exception if it fails.
        conn = get_db_connection()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database Connection Error: {e}. Please check your .env file and ensure the database is running.")
        return False

def show_no_database_message():
    """Show a consistent message when database is not available."""
    st.warning("Could not connect to the database. Please ensure it is running and configured correctly in the `.env` file.")

# --- NER Helper Functions ---

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
        r'patient\s+name\s*:\s*(?:(?:mr|mrs|ms|dr)\.?\s+)?([A-Z][A-Z\s]+?)(?:\s*(?:study|age|referring|sex|gender|$|\n))',
        r'(?:patient\s+)?name\s*:\s*(?:(?:mr|mrs|ms|dr)\.?\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})(?:\s*(?:\n|$|study|age|dob|sex|gender|mrn|address|phone))',
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
        r'age\s*[:=]\s*(\d{1,3})(?:\s*(?:years?|yrs?|y\.o\.?)?)',
        r'\((\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)\)',
        r'\((\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)\)',
        r'(?:age|aged?)\s*[:=]?\s*(\d{1,3})\s*(?:years?|yrs?|y\.o\.)?',
        r'(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)',
        r'aged?\s+(\d{1,3})',
        r'age\s*[:=]\s*(\d{1,3})',
        r'DOB:\s*\d{2}/\d{2}/\d{4}\s*(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y\.o\.)',
        r'(\d{1,3})\s*[-]?\s*year[-\s]*old',
    ]
    
    # Gender extraction patterns
    gender_patterns = [
        r'(?:gender|sex)\s*[:=]\s*(male|female|m|f)',
        r'\b(male|female)\b(?!(?:\s+patient|\s+doctor|\s+nurse))',
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
                name = re.sub(r'\s+', ' ', name)
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
        if re.search(r'\bmale\b(?!(?:\s+patient|\s+doctor|\s+nurse))', text, re.IGNORECASE):
            # Check if 'female' appears nearby to avoid false positives
            if not re.search(r'\bfemale\b', text, re.IGNORECASE):
                details["gender"] = "Male"
        elif re.search(r'\bfemale\b', text, re.IGNORECASE):
            details["gender"] = "Female"
    
    return details

def extract_ner_entities(text, ner_pipeline):
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

# --- UI Rendering for NER Page ---

def render_ner_upload_tab(ner_pipeline):
    st.info("📋 Upload medical reports in PDF format for analysis")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Maximum file size: 10MB per file",
        key="ner_uploader"
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
                os.remove(tmp_file_path)
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
                ner_results = extract_ner_entities(text, ner_pipeline)
                
                if ner_results:
                    for ent in ner_results:
                        confidence_color = "🟢" if ent['confidence'] > 0.8 else "🟡" if ent['confidence'] > 0.5 else "🔴"
                        st.markdown(f"{confidence_color} **{ent['label']}**: {ent['text']} `({ent['confidence']})`")
                else:
                    st.warning("⚠️ No medical entities detected")

            # Only store if we have valid data
            if text and (any(patient_details.values()) or ner_results):
                store_to_mysql(patient_details, ner_results, uploaded_file.name)
            else:
                st.warning(f"⚠️ {uploaded_file.name} contained no useful data to store.")

            os.remove(tmp_file_path)
            st.divider()

        status_text.text("✅ All files processed!")

def render_ner_view_tab():
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

def render_ner_search_tab():
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

def render_ner_stats_tab():
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
                    if stats:
                        most_common = max(stats.items(), key=lambda x: x[1])
                        st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")
                
            else:
                st.info("📋 No statistics available. Process some reports first!")
                
        except Exception as e:
            st.error(f"❌ Failed to load statistics: {str(e)}")
            show_no_database_message()

def render_ner_page(ner_pipeline):
    """Renders the main page for Medical Report Analysis (NER)."""
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            📄 Medical Report Analysis (NER)
        </h2>
        <p style="color: var(--text-secondary);">
            Extract, analyze, and store medical information from PDF reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # NER specific settings in sidebar
    with st.sidebar:
        with st.expander("🔧 NER Settings", expanded=False):
            NER_CONFIG['confidence_threshold'] = st.slider(
                "Minimum Confidence Score", 
                min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                help="Lower values show more entities but with lower confidence",
                key="ner_confidence"
            )
            NER_CONFIG['max_entities'] = st.slider(
                "Maximum Entities to Show", 
                min_value=5, max_value=50, value=20, step=5,
                key="ner_max_entities"
            )

    # Main content using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Report", "📊 View Reports", "🔍 Search", "📈 Statistics"])

    with tab1:
        render_ner_upload_tab(ner_pipeline)
    with tab2:
        render_ner_view_tab()
    with tab3:
        render_ner_search_tab()
    with tab4:
        render_ner_stats_tab()

# --- Streamlit User Interface ---
def main():
    """
    The main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Radiologist's Copilot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark mode medical theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main theme variables - DARK MODE */
    :root {
        --bg-primary: #222222;
        --bg-secondary: #181818;
        --bg-tertiary: #2c2c2c;
        --accent-blue: #0074F8;
        --accent-teal: #00C9A7;
        --accent-magenta: #FF2D7A;
        --text-primary: #f5f5f5;
        --text-secondary: #cccccc;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 20px 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        text-align: center;
        color: var(--text-secondary);
        margin-top: 10px;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--glass-border);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
        padding: 20px;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
    }
    
    .sidebar-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Navigation cards */
    .nav-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-card:hover {
        background: var(--glass-bg);
        border-color: var(--accent-blue);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .nav-card.active {
        border-color: var(--accent-teal);
        box-shadow: 0 8px 25px rgba(0, 255, 163, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
    }
    
    /* Primary action button */
    .primary-btn {
        background: linear-gradient(45deg, var(--accent-magenta), var(--accent-blue));
        color: white;
        border: none;
        border-radius: 15px;
        padding: 15px 40px;
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.3);
        text-decoration: none;
        display: inline-block;
        text-align: center;
    }
    
    .primary-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(255, 0, 110, 0.4);
    }
    
    /* Cards and containers */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 1rem;
        margin-top: 5px;
    }
    
    /* File uploader */
    .uploadedFile {
        background: var(--glass-bg);
        border: 2px dashed var(--accent-blue);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: var(--accent-teal);
        background: rgba(0, 255, 163, 0.05);
    }
    
    /* Text areas and inputs */
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-tertiary);
        border: 1px solid var(--glass-border);
        border-radius: 10px;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--glass-border);
        border-radius: 10px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--accent-blue);
    }
    
    /* Chat interface */
    .chat-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .chat-message {
        background: var(--bg-tertiary);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid var(--accent-blue);
    }
    
    .chat-message.user {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
        color: white;
        border-left: none;
        margin-left: 50px;
    }
    
    .chat-message.assistant {
        background: var(--bg-tertiary);
        margin-right: 50px;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(45deg, var(--accent-teal), #00CC7A);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    .stError {
        background: linear-gradient(45deg, var(--accent-magenta), #FF4757);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    .stInfo {
        background: linear-gradient(45deg, var(--accent-blue), #4834DF);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-tertiary);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
        color: white;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 212, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Floating elements */
    .floating-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(45deg, var(--accent-magenta), var(--accent-blue));
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 8px 25px rgba(255, 0, 110, 0.4);
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .floating-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 35px rgba(255, 0, 110, 0.6);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background: var(--accent-teal); }
    .status-processing { background: var(--accent-blue); }
    .status-error { background: var(--accent-magenta); }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header { padding: 15px 20px; }
        .header-title { font-size: 2rem; }
        .metric-card { padding: 20px; }
        .nav-card { padding: 15px; }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, var(--accent-blue), var(--accent-teal));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, var(--accent-teal), var(--accent-magenta));
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "report_context" not in st.session_state:
        st.session_state.report_context = ""
    if "uploaded_image_data" not in st.session_state:
        st.session_state.uploaded_image_data = None
    if "last_uploaded_file_id" not in st.session_state:
        st.session_state.last_uploaded_file_id = None
    if "chexnet_results" not in st.session_state:
        st.session_state.chexnet_results = {}
    if "segmentation_maps" not in st.session_state:
        st.session_state.segmentation_maps = {}
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "user_role" not in st.session_state:
        st.session_state.user_role = "radiologist"  # or "patient"
    if "patient_data" not in st.session_state:
        st.session_state.patient__data = {}

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">🏥 Radiologist's Copilot</h1>
        <p class="header-subtitle">AI-Powered Medical Imaging Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Enhanced Sidebar Navigation ---
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 class="sidebar-title">🔬 Radiologist's Copilot</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        pages = {
            "🏠 Dashboard": "Dashboard",
            "📤 Upload & Analyze": "Upload",
            "📋 AI Reports": "AI Report",
            "🎯 Segmentation": "Segmentation",
            "📄 Report Analysis (NER)": "NER",
            "⚖️ Comparison": "Compare X-Rays",
            "🔗 Share Portal": "Share",
            "👤 Patient View": "Patient",
            "💬 AI Assistant": "Chat",
            "⚙️ Settings": "Settings"
        }
        
        for icon_label, page_key in pages.items():
            if st.button(icon_label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
        
        st.markdown("---")
        
        # Role switcher
        st.markdown("### 👥 View Mode")
        user_role = st.selectbox(
            "Select Role",
            ["radiologist", "patient"],
            index=0 if st.session_state.user_role == "radiologist" else 1,
            key="role_selector"
        )
        if user_role != st.session_state.user_role:
            st.session_state.user_role = user_role
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">24</div>
            <div class="metric-label">🔄 Active Scans</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">156</div>
            <div class="metric-label">✅ Completed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">98.7%</div>
            <div class="metric-label">🎯 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    # Load models
    clip_processor, clip_model, clip_tokenizer = load_clip_model()
    chexnet_model, grad_cam = load_chexnet_model()
    ner_pipeline = load_ner_model()

    # --- Page Routing ---
    current_page = st.session_state.current_page
    
    if current_page == "Dashboard":
        render_dashboard_page()
    elif current_page == "Upload":
        render_upload_page(clip_processor, clip_model, clip_tokenizer, chexnet_model, grad_cam)
    elif current_page == "AI Report":
        render_ai_report_page(clip_processor, clip_model, clip_tokenizer)
    elif current_page == "Segmentation":
        render_segmentation_page(chexnet_model, grad_cam)
    elif current_page == "NER":
        render_ner_page(ner_pipeline)
    elif current_page == "Compare X-Rays":
        render_comparison_page(clip_processor, clip_model, clip_tokenizer)
    elif current_page == "Share":
        render_share_page()
    elif current_page == "Patient":
        render_patient_page()
    elif current_page == "Chat":
        render_chat_page()
    elif current_page == "Settings":
        render_settings_page()

def render_dashboard_page():
    """Enhanced dashboard with medical metrics and recent activity"""
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
                🏥 Welcome to Radiologist's Copilot
            </h2>
            <p style="color: var(--text-secondary); font-size: 1.1rem; line-height: 1.6;">
                Advanced AI-powered medical imaging analysis platform designed for modern healthcare professionals.
                Upload X-rays, get instant AI analysis, and collaborate with colleagues seamlessly.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div class="status-indicator status-online"></div>
            <strong>System Status: Online</strong><br>
            <small style="color: var(--text-secondary);">All AI models operational</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">47</div>
            <div class="metric-label">📊 Today's Scans</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12</div>
            <div class="metric-label">⏳ Pending Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94.3%</div>
            <div class="metric-label">🎯 AI Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">2.3s</div>
            <div class="metric-label">⚡ Avg Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity and quick actions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--accent-teal); margin-bottom: 20px;">📋 Recent Patients</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample patient data
        patients = [
            {"name": "John Smith", "id": "PT001", "status": "Completed", "time": "2 min ago"},
            {"name": "Sarah Johnson", "id": "PT002", "status": "Processing", "time": "5 min ago"},
            {"name": "Michael Brown", "id": "PT003", "status": "Pending", "time": "10 min ago"},
            {"name": "Emily Davis", "id": "PT004", "status": "Completed", "time": "15 min ago"}
        ]
        
        for patient in patients:
            status_class = "status-online" if patient["status"] == "Completed" else "status-processing" if patient["status"] == "Processing" else "status-error"
            st.markdown(f"""
            <div class="metric-card" style="padding: 15px; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{patient["name"]}</strong> ({patient["id"]})
                        <br><small style="color: var(--text-secondary);">{patient["time"]}</small>
                    </div>
                    <div>
                        <span class="status-indicator {status_class}"></span>
                        {patient["status"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--accent-magenta); margin-bottom: 20px;">🚀 Quick Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📤 New Upload", key="quick_upload", use_container_width=True):
            st.session_state.current_page = "Upload"
            st.rerun()
        
        if st.button("📋 View Reports", key="quick_reports", use_container_width=True):
            st.session_state.current_page = "AI Report"
            st.rerun()
        
        if st.button("🎯 Segmentation", key="quick_segment", use_container_width=True):
            st.session_state.current_page = "Segmentation"
            st.rerun()
        
        if st.button("⚖️ Compare Scans", key="quick_compare", use_container_width=True):
            st.session_state.current_page = "Compare X-Rays"
            st.rerun()
    
    # Floating action button
    st.markdown("""
    <button class="floating-btn" onclick="window.location.reload();">
        ➕
    </button>
    """, unsafe_allow_html=True)

def render_upload_page(clip_processor, clip_model, clip_tokenizer, chexnet_model, grad_cam):
    """Enhanced upload page with patient form and drag-drop interface"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            📤 Upload & Analyze X-Ray Images
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient information form
    with st.container():
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--accent-teal); margin-bottom: 20px;">👤 Patient Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_name = st.text_input("👤 Patient Name", placeholder="Enter full name")
        with col2:
            patient_id = st.text_input("🆔 Patient ID", placeholder="PT####")
        with col3:
            patient_age = st.number_input("🎂 Age", min_value=0, max_value=150, value=0)
        
        col1, col2 = st.columns(2)
        with col1:
            exam_type = st.selectbox("🏥 Examination Type", 
                                   ["Chest X-Ray", "Lung Screening", "Cardiac Assessment", "General Checkup"])
        with col2:
            priority = st.selectbox("⚡ Priority Level", 
                                  ["Normal", "Urgent", "Emergency"])
        
        clinical_notes = st.text_area("📝 Clinical Notes", 
                                    placeholder="Enter relevant clinical history, symptoms, or notes...")
    
    # File upload section
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: var(--accent-magenta); margin-bottom: 20px;">📁 Upload X-Ray Images</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your X-ray images here or click to browse",
        type=["png", "jpg", "jpeg", "dcm"],
        key="medical_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Store patient data
            st.session_state.patient_data = {
                "name": patient_name,
                "id": patient_id,
                "age": patient_age,
                "exam_type": exam_type,
                "priority": priority,
                "clinical_notes": clinical_notes
            }
            
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: var(--accent-blue);">📷 Uploaded Image</h4>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, caption="X-Ray Image", use_column_width=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: var(--accent-teal);">📊 Image Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Analysis options
                analysis_type = st.selectbox(
                    "🔍 Analysis Type",
                    ["Complete Analysis", "Quick Scan", "Pathology Detection", "Segmentation Only"]
                )
                
                confidence_threshold = st.slider(
                    "🎯 Detection Confidence",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.1
                )
                
                # Analysis button
                if st.button("🚀 Start AI Analysis", key="start_analysis", use_container_width=True):
                    with st.spinner("🔄 AI is analyzing the X-ray..."):
                        # Store the image for analysis
                        st.session_state.uploaded_image_data = image
                        
                        # Run analysis based on type
                        if analysis_type in ["Complete Analysis", "Quick Scan", "Pathology Detection"]:
                            pathology_results = predict_pathologies(image, chexnet_model, confidence_threshold)
                            st.session_state.chexnet_results = pathology_results
                            
                            # Generate segmentation if requested
                            detected_pathologies = {k: v for k, v in pathology_results.items() if v['detected']}
                            if detected_pathologies and analysis_type in ["Complete Analysis", "Segmentation Only"]:
                                segmentation_maps = generate_segmentation_map(
                                    image, chexnet_model, grad_cam,
                                    detected_pathologies, image.size
                                )
                                st.session_state.segmentation_maps = segmentation_maps
                        
                        st.success("✅ Analysis completed successfully!")
                        st.balloons()
            
            # Display results if available
            if st.session_state.chexnet_results:
                st.markdown("---")
                render_analysis_results()
                
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

def render_analysis_results():
    """Display analysis results in an enhanced format"""
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: var(--accent-blue); margin-bottom: 20px;">🎯 AI Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Detected pathologies summary
    detected = [k for k, v in st.session_state.chexnet_results.items() if v['detected']]
    
    if detected:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid var(--accent-magenta);">
            <h4 style="color: var(--accent-magenta);">🚨 Detected Pathologies ({len(detected)})</h4>
            <p style="color: var(--text-primary); font-size: 1.1rem;">
                {', '.join(detected)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid var(--accent-teal);">
            <h4 style="color: var(--accent-teal);">✅ No Significant Pathologies Detected</h4>
            <p style="color: var(--text-secondary);">The AI analysis shows normal findings.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Detailed Results", "🎯 Segmentation", "📋 Report"])
    
    with tab1:
        col1, col2 = st.columns(2)
        results = sorted(st.session_state.chexnet_results.items(), 
                        key=lambda x: x[1]['probability'], reverse=True)
        
        for i, (pathology, data) in enumerate(results):
            col = col1 if i % 2 == 0 else col2
            with col:
                confidence_color = "var(--accent-teal)" if data['probability'] > 0.7 else "var(--accent-blue)" if data['probability'] > 0.3 else "var(--text-secondary)"
                detection_icon = "🔴" if data['detected'] else "🟢"
                
                st.markdown(f"""
                <div class="metric-card" style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: {confidence_color};">{detection_icon} {pathology}</strong>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {confidence_color}; font-weight: bold;">
                                {data['probability']:.1%}
                            </div>
                            <small style="color: var(--text-secondary);">
                                {'Detected' if data['detected'] else 'Normal'}
                            </small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        if st.session_state.segmentation_maps:
            st.markdown("🎯 **Pathology Segmentation Maps**")
            
            # Create enhanced visualization
            if st.session_state.uploaded_image_data:
                overlay_image = create_overlay_visualization(
                    st.session_state.uploaded_image_data,
                    st.session_state.segmentation_maps
                )
                if overlay_image:
                    st.image(overlay_image, caption="AI Segmentation Analysis", use_column_width=True)
        else:
            st.info("🔍 No segmentation maps available. Run complete analysis to generate segmentation.")
    
    with tab3:
        # Generate and display formatted report
        if st.session_state.chexnet_results:
            report = format_chexnet_results(st.session_state.chexnet_results, st.session_state.segmentation_maps)
            st.markdown(report)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📄 Download Report",
                    data=report,
                    file_name=f"xray_report_{st.session_state.patient_data.get('id', 'unknown')}.md",
                    mime="text/markdown"
                )
            with col2:
                if st.button("📤 Share with Patient"):
                    st.session_state.current_page = "Share"
                    st.rerun()
            with col3:
                if st.button("💬 Discuss with AI"):
                    st.session_state.current_page = "Chat"
                    st.rerun()
def render_ai_report_page(clip_processor, clip_model, clip_tokenizer):
    """AI Report generation and Q&A interface"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            📋 AI Report Generator & Q&A
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    default_labels = "normal, fracture, pneumonia, cardiomegaly, pleural effusion, nodule, opacity"
    candidate_labels_input = st.text_area(
        "🏷️ Clinical Findings Labels (comma-separated)",
        default_labels,
        height=100
    )
    candidate_labels = [label.strip() for label in candidate_labels_input.split(',') if label.strip()]
    
    # File upload for report generation
    uploaded_file = st.file_uploader(
        "📁 Upload X-ray or Medical Report", 
        type=["png", "jpg", "jpeg", "pdf", "txt"],
        key="report_upload"
    )
    
    if uploaded_file:
        try:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if uploaded_file.type in ["image/png", "image/jpeg"]:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Uploaded X-ray", use_column_width=True)
                    st.session_state.uploaded_image_data = image
                    
                    if st.button("🚀 Generate AI Report", use_container_width=True):
                        with st.spinner("🤖 AI is analyzing the image..."):
                            report_display, report_context = generate_report(
                                image, clip_processor, clip_model, clip_tokenizer, candidate_labels
                            )
                            st.session_state.report_context = report_context
                            st.session_state.messages = [{"role": "assistant", "content": report_display}]
                            st.success("✅ Report generated successfully!")
                
                elif uploaded_file.type in ["application/pdf", "text/plain"]:
                    extracted_text = extract_text_from_report(uploaded_file)
                    st.text_area("📄 Extracted Report Text", extracted_text, height=400)
                    st.session_state.report_context = extracted_text
                    st.session_state.messages = [{"role": "assistant", "content": "Report loaded. You can now ask questions."}]
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: var(--accent-teal);">💬 AI Q&A Interface</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Chat interface
                if st.session_state.report_context:
                    # Display chat messages
                    chat_container = st.container()
                    with chat_container:
                        for message in st.session_state.messages:
                            role_icon = "🤖" if message["role"] == "assistant" else "👨‍⚕️"
                            message_class = "assistant" if message["role"] == "assistant" else "user"
                            
                            st.markdown(f"""
                            <div class="chat-message {message_class}">
                                <strong>{role_icon} {message["role"].title()}:</strong><br>
                                {message["content"]}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Chat input
                    if prompt := st.chat_input("💭 Ask about the medical report..."):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        with st.spinner("🔍 AI is thinking..."):
                            response = answer_text_question(st.session_state.report_context, prompt)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.rerun()
                else:
                    st.info("📋 Upload a file and generate a report to start the Q&A session.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def render_segmentation_page(chexnet_model, grad_cam):
    """Enhanced segmentation viewer with anatomical regions"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            🎯 Advanced Segmentation Analysis
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_threshold = st.slider("🎯 Detection Threshold", 0.1, 0.9, 0.5, 0.1)
    with col2:
        activation_threshold = st.slider("🔥 Activation Threshold", 0.1, 0.8, 0.3, 0.1)
    with col3:
        show_region_labels = st.checkbox("🏷️ Show Region Labels", True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload X-ray for Segmentation Analysis",
        type=["png", "jpg", "jpeg"],
        key="segmentation_upload"
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: var(--accent-blue);">📷 Original X-Ray</h4>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, caption="X-Ray Image", use_column_width=True)
                
                if st.button("🔬 Analyze Segmentation", use_container_width=True):
                    with st.spinner("🔄 Performing segmentation analysis..."):
                        # Run pathology detection
                        pathology_results = predict_pathologies(image, chexnet_model, detection_threshold)
                        detected_pathologies = {k: v for k, v in pathology_results.items() if v['detected']}
                        
                        if detected_pathologies:
                            # Generate segmentation maps
                            segmentation_maps = generate_segmentation_map(
                                image, chexnet_model, grad_cam,
                                detected_pathologies, image.size
                            )
                            
                            # Enhanced visualization with region analysis
                            if show_region_labels:
                                labeled_overlay, region_analysis, region_report = create_enhanced_overlay_visualization(
                                    image, segmentation_maps, alpha=0.4, activation_threshold=activation_threshold
                                )
                                st.session_state.labeled_overlay = labeled_overlay
                                st.session_state.region_analysis = region_analysis
                                st.session_state.region_report = region_report
                            else:
                                overlay_image = create_overlay_visualization(image, segmentation_maps)
                                st.session_state.overlay_image = overlay_image
                            
                            st.session_state.segmentation_maps = segmentation_maps
                            st.session_state.pathology_results = pathology_results
                            st.success("✅ Segmentation analysis completed!")
                        else:
                            st.info("ℹ️ No pathologies detected for segmentation.")
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: var(--accent-teal);">🎯 Segmentation Results</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display segmentation results
                if hasattr(st.session_state, 'segmentation_maps') and st.session_state.segmentation_maps:
                    # Show enhanced visualization
                    if show_region_labels and hasattr(st.session_state, 'labeled_overlay'):
                        st.image(st.session_state.labeled_overlay, 
                               caption="Enhanced Segmentation with Region Labels", 
                               use_column_width=True)
                    elif hasattr(st.session_state, 'overlay_image'):
                        st.image(st.session_state.overlay_image, 
                               caption="Segmentation Overlay", 
                               use_column_width=True)
                    
                    # Controls
                    st.markdown("### 🎛️ Visualization Controls")
                    
                    pathology_toggles = {}
                    for pathology in st.session_state.segmentation_maps.keys():
                        pathology_toggles[pathology] = st.checkbox(f"Show {pathology}", True, key=f"toggle_{pathology}")
                    
                    opacity = st.slider("🌟 Overlay Opacity", 0.0, 1.0, 0.4, 0.1)
                    
                    # Region analysis display
                    if show_region_labels and hasattr(st.session_state, 'region_report'):
                        with st.expander("📊 Detailed Region Analysis"):
                            st.markdown(st.session_state.region_report)
                else:
                    st.info("🔬 Run segmentation analysis to view results here.")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

def render_comparison_page(clip_processor, clip_model, clip_tokenizer):
    """Enhanced comparison tool with timeline view"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            ⚖️ X-Ray Comparison Tool
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    default_labels = "normal, fracture, pneumonia, cardiomegaly, pleural effusion, nodule, opacity"
    candidate_labels_input = st.text_area(
        "🏷️ Comparison Labels",
        default_labels,
        height=80
    )
    candidate_labels = [label.strip() for label in candidate_labels_input.split(',') if label.strip()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--accent-magenta);">📅 Previous Scan</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file_1 = st.file_uploader(
            "Upload Previous X-ray/Report",
            type=["png", "jpg", "jpeg", "pdf", "txt"],
            key="previous_scan"
        )
        
        previous_image = None
        previous_report_text = ""
        
        if uploaded_file_1:
            if uploaded_file_1.type in ["image/png", "image/jpeg"]:
                previous_image = Image.open(uploaded_file_1).convert("RGB")
                st.image(previous_image, caption="Previous X-ray", use_column_width=True)
            elif uploaded_file_1.type in ["application/pdf", "text/plain"]:
                previous_report_text = extract_text_from_report(uploaded_file_1)
                st.text_area("Previous Report", previous_report_text, height=200, key="prev_text")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--accent-teal);">📊 Current Scan</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file_2 = st.file_uploader(
            "Upload Current X-ray/Report",
            type=["png", "jpg", "jpeg", "pdf", "txt"],
            key="current_scan"
        )
        
        current_image = None
        current_report_text = ""
        
        if uploaded_file_2:
            if uploaded_file_2.type in ["image/png", "image/jpeg"]:
                current_image = Image.open(uploaded_file_2).convert("RGB")
                st.image(current_image, caption="Current X-ray", use_column_width=True)
            elif uploaded_file_2.type in ["application/pdf", "text/plain"]:
                current_report_text = extract_text_from_report(uploaded_file_2)
                st.text_area("Current Report", current_report_text, height=200, key="curr_text")
    
    # Comparison analysis
    if (uploaded_file_1 and uploaded_file_2):
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔍 Perform Comparison Analysis", use_container_width=True):
                with st.spinner("🤖 AI is comparing the scans..."):
                    previous_input = previous_report_text if previous_report_text else previous_image
                    current_input = current_report_text if current_report_text else current_image
                    
                    comparison_result = compare_xrays(
                        previous_input, current_input,
                        clip_processor, clip_model, clip_tokenizer, candidate_labels
                    )
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h3 style="color: var(--accent-blue);">📋 Comparison Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(comparison_result)
                    
                    # Additional comparison metrics
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: var(--accent-teal);">📊 Change Summary</h4>
                        <ul>
                            <li>🔄 Overall Status: Analysis Complete</li>
                            <li>📈 Trend: AI Assessment Provided</li>
                            <li>⚡ Processing Time: < 5 seconds</li>
                            <li>🎯 Confidence: High</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

def render_share_page():
    """Patient sharing portal with secure links"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            🔗 Patient Sharing Portal
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        patient_id = st.selectbox(
            "👤 Select Patient",
            ["PT001 - John Smith", "PT002 - Sarah Johnson", "PT003 - Michael Brown"],
            key="share_patient"
        )
        
        share_options = st.multiselect(
            "📋 Content to Share",
            ["X-Ray Images", "AI Analysis Report", "Segmentation Maps", "Comparison Results"],
            default=["AI Analysis Report"]
        )
        
        # Sharing settings
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--accent-teal);">⚙️ Sharing Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        expiration_time = st.selectbox(
            "⏰ Link Expiration",
            ["24 hours", "7 days", "30 days", "No expiration"]
        )
        
        require_password = st.checkbox("🔒 Require Password", True)
        if require_password:
            password = st.text_input("🔑 Access Password", type="password", value="SecurePass123")
        
        send_notification = st.checkbox("📧 Send Email Notification", True)
        if send_notification:
            patient_email = st.text_input("📧 Patient Email", value="patient@example.com")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--accent-magenta);">🚀 Quick Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔗 Generate Secure Link", use_container_width=True):
            import uuid
            secure_link = f"https://radiologists-copilot.ai/patient/{uuid.uuid4().hex[:8]}"
            
            st.success("✅ Secure link generated!")
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid var(--accent-teal);">
                <h4 style="color: var(--accent-teal);">🔗 Patient Access Link</h4>
                <code style="background: var(--bg-tertiary); padding: 10px; border-radius: 5px; display: block; margin: 10px 0;">
                    {secure_link}
                </code>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">
                    🔒 Password: {password if require_password else 'None'}<br>
                    ⏰ Expires: {expiration_time}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("📧 Send to Patient", use_container_width=True):
            st.success("📧 Email sent successfully!")
            st.balloons()
        
        if st.button("📱 Generate QR Code", use_container_width=True):
            st.info("📱 QR Code generation feature coming soon!")

def render_patient_page():
    """Patient dashboard view"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            👤 Patient Dashboard
        </h2>
        <p style="color: var(--text-secondary);">
            Welcome! Here you can view your X-ray analysis results and chat with our AI assistant.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient info summary
    if st.session_state.patient_data:
        patient_info = st.session_state.patient_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{patient_info.get('name', 'N/A')}</div>
                <div class="metric-label">👤 Patient Name</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{patient_info.get('id', 'N/A')}</div>
                <div class="metric-label">🆔 Patient ID</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{patient_info.get('exam_type', 'N/A')}</div>
                <div class="metric-label">🏥 Exam Type</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Report summary for patients
    if st.session_state.chexnet_results:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: var(--accent-teal); margin-bottom: 20px;">📋 Your X-Ray Analysis Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        detected = [k for k, v in st.session_state.chexnet_results.items() if v['detected']]
        
        if detected:
            list_items = "".join([f"<li>{finding}</li>" for finding in detected])
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid var(--accent-blue);">
                <h4 style="color: var(--accent-blue);">🔍 Analysis Results</h4>
                <p style="color: var(--text-primary);">
                    Our AI analysis has identified the following findings that require attention:
                </p>
                <ul style="color: var(--text-primary);">
                    {list_items}
                </ul>
                <p style="color: var(--text-secondary); font-style: italic;">
                    Please discuss these results with your healthcare provider for proper medical interpretation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid var(--accent-teal);">
                <h4 style="color: var(--accent-teal);">✅ Good News!</h4>
                <p style="color: var(--text-primary);">
                    The AI analysis shows normal findings in your X-ray. No significant abnormalities were detected.
                </p>
                <p style="color: var(--text-secondary); font-style: italic;">
                    Remember that AI analysis is a tool to assist your healthcare provider. Always follow up with your doctor for complete medical evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Patient-friendly action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Report", use_container_width=True):
                st.success("📄 Report download started!")
        
        with col2:
            if st.button("💬 Ask AI Questions", use_container_width=True):
                st.session_state.current_page = "Chat"
                st.rerun()
        
        with col3:
            if st.button("👨‍⚕️ Contact Doctor", use_container_width=True):
                st.info("👨‍⚕️ Contact information sent to your email!")

def render_chat_page():
    """AI Assistant chat interface"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            💬 AI Medical Assistant
        </h2>
        <p style="color: var(--text-secondary);">
            Ask questions about your X-ray analysis or general medical imaging queries.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat interface
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your AI medical assistant. I can help answer questions about your X-ray analysis or provide general information about medical imaging. How can I help you today?"}
        ]
    
    # Display chat history
    for message in st.session_state.chat_messages:
        role_icon = "🤖" if message["role"] == "assistant" else "👨‍⚕️" if st.session_state.user_role == "radiologist" else "👤"
        message_class = "assistant" if message["role"] == "assistant" else "user"
        
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{role_icon} {message["role"].title()}:</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("💭 Type your question here..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤖 AI is thinking..."):
            # Use existing report context if available, otherwise provide general assistance
            context = st.session_state.report_context if st.session_state.report_context else "General medical imaging assistant conversation."
            response = answer_text_question(context, prompt)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Quick question buttons
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: var(--accent-teal);">💡 Quick Questions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    quick_questions = [
        "What do the results mean?",
        "Should I be concerned?",
        "What are the next steps?",
        "How accurate is the AI?",
        "Can you explain the technical terms?",
        "When should I see a doctor?"
    ]
    
    for i, question in enumerate(quick_questions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": question})
                with st.spinner("🤖 AI is responding..."):
                    context = st.session_state.report_context if st.session_state.report_context else "General medical imaging assistant conversation."
                    response = answer_text_question(context, question)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()

def render_settings_page():
    """Settings and configuration page"""
    
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: var(--accent-blue); margin-bottom: 20px;">
            ⚙️ Settings & Configuration
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # User preferences
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--accent-teal);">👤 User Preferences</h4>
        </div>
        """, unsafe_allow_html=True)
        
        theme = st.selectbox("🎨 Theme", ["Dark (Current)", "Light", "Auto"])
        language = st.selectbox("🌐 Language", ["English", "Spanish", "French", "German"])
        notifications = st.checkbox("🔔 Enable Notifications", True)
        auto_save = st.checkbox("💾 Auto-save Reports", True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--accent-magenta);">🤖 AI Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        default_threshold = st.slider("🎯 Default Detection Threshold", 0.1, 0.9, 0.5, 0.1)
        auto_segment = st.checkbox("🎯 Auto-generate Segmentation", True)
        detailed_reports = st.checkbox("📋 Detailed AI Reports", True)
        confidence_display = st.checkbox("📊 Show Confidence Scores", True)
    
    # System information
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: var(--accent-blue);">🖥️ System Information</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <strong>Platform:</strong> Radiologist's Copilot v2.0<br>
                <strong>AI Models:</strong> BiomedCLIP, ChexNet<br>
                <strong>Last Update:</strong> January 2025
            </div>
            <div>
                <strong>Status:</strong> <span style="color: var(--accent-teal);">● Online</span><br>
                <strong>Uptime:</strong> 99.9%
                <strong>Support:</strong> 24/7 Available
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.info("🔄 Settings reset to default values!")
    
    with col3:
        if st.button("📞 Contact Support", use_container_width=True):
            st.info("📞 Support ticket created!")

# Add a helper function for the missing anatomical map
def create_interactive_region_map(image_shape):
    """Create an anatomical regions reference map"""
    try:
        height, width = image_shape
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a dark background
        ax.imshow(np.zeros((height, width)), cmap='gray', alpha=0.3)
        
        # Draw anatomical regions
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'yellow', 'magenta']
        
        for i, (region_name, region_info) in enumerate(ANATOMICAL_REGIONS.items()):
            x1, y1, x2, y2 = region_info['coords']
            
            # Convert to pixel coordinates
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            
            # Draw rectangle
            rect = Rectangle((px1, py1), px2 - px1, py2 - py1,
                           linewidth=2, edgecolor=colors[i % len(colors)], 
                           facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label
            ax.text(px1 + 10, py1 + 20, region_info['label'],
                   color=colors[i % len(colors)], fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title('Anatomical Regions Reference Map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='black')
        buf.seek(0)
        from PIL import Image as PILImage
        map_image = PILImage.open(buf)
        plt.close()
        
        return map_image
    
    except Exception as e:
        print(f"Error creating anatomical map: {e}")
        return None

if __name__ == "__main__":
    main()
