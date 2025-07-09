# Sahayak AI Education Platform - Enhanced UI Version
# Import required libraries
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import re
import torch
import traceback
from typing import Dict, List, Any
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import pytesseract
import time
import requests
from playwright.sync_api import sync_playwright

# Import prompt templates
from prompts import (
    get_test_creation_prompt,
    get_test_creation_retry_prompt,
    get_hint_generator_prompt,
    get_grading_prompt,
    get_analytics_prompt
)

# ------------------------ Setup ------------------------ #
# Configure page settings
st.set_page_config(
    page_title="📚 Sahayak: AI Education Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Check for required API key
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# Initialize translator
translator = GoogleTranslator(source="auto", target="en")

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    """Load and cache AI models"""
    models = {}
    try:
        # Initialize LLM
        models['llm'] = ChatGroq(
            model_name="gemma2-9b-it",
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        models['llm'] = None
    
    try:
        # Initialize image processing models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        models['processor'] = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=dtype
        ).to(device)
        models['blip_model'].eval()
    except Exception as e:
        st.error(f"Failed to load image processor: {e}")
        models['processor'] = None
        models['blip_model'] = None
    
    return models

# Load models
models = load_models()

# ------------------------ Session State ------------------------ #
# Initialize session state variables
if "test_history" not in st.session_state:
    st.session_state.test_history = []
if "student_profiles" not in st.session_state:
    st.session_state.student_profiles = {}
if "last_results" not in st.session_state:
    st.session_state.last_results = {}
if "class_level" not in st.session_state:
    st.session_state.class_level = "Class 5"
if "language" not in st.session_state:
    st.session_state.language = "English"

# ------------------------ Helper Functions ------------------------ #
def query_langchain(prompt: str, retry: bool = False) -> str:
    """Query the language model with a prompt"""
    if not models['llm']:
        return "Error: LLM service unavailable"
    try:
        response = models['llm'].invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"

def process_image(image: Image.Image, use_ocr: bool = True) -> str:
    """Process an image to extract text or generate description"""
    if use_ocr:
        try:
            text = pytesseract.image_to_string(
                image,
                lang="eng+hin+tam+ben+tel+mar+guj+kan+mal+pan"
            )
            if text.strip():
                return text.strip()
            st.warning("OCR extracted no text. Falling back to BLIP.")
        except Exception as e:
            st.warning(f"OCR failed: {str(e)}. Falling back to BLIP.")
    
    if not models['processor'] or not models['blip_model']:
        return "Error: Image processing unavailable"
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        device = next(models['blip_model'].parameters()).device
        inputs = models['processor'](images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = models['blip_model'].generate(**inputs, max_new_tokens=100)
        description = models['processor'].decode(outputs[0], skip_special_tokens=True)
        return description if description else "Error: No content extracted from image"
    except Exception as e:
        return f"Error: Image processing failed - {str(e)}"

def process_url(url: str) -> str:
    """Extract content from a URL"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=10000)
            page.wait_for_load_state("domcontentloaded")
            content = page.content()
            text = page.evaluate(
                """() => {
                    document.querySelectorAll('script, style, nav, header, footer').forEach(el => el.remove());
                    return document.body.innerText.trim();
                }"""
            )
            browser.close()
        return text[:5000] if text else "Error: No content extracted from URL"
    except Exception as e:
        st.error(f"Failed to fetch URL content: {str(e)}")
        return f"Error: URL processing failed - {str(e)}"

def extract_questions_and_answers(text: str) -> List[Dict[str, Any]]:
    """Parse generated text into structured questions"""
    questions = []
    question_blocks = re.split(r'Question \d+:', text)[1:]
    
    for block in question_blocks:
        question_text_match = re.search(r'^(.*?)(?:\n|$)', block)
        question_text = question_text_match.group(1).strip() if question_text_match else "Not found"
        
        type_match = re.search(r'Type:\s*(.*)', block)
        q_type = type_match.group(1).strip() if type_match else "MCQ"
        
        options_match = re.search(r'Options:\s*(.*)', block)
        options = [opt.strip() for opt in options_match.group(1).split(";")] if options_match else []
        
        answer_match = re.search(r'Correct Answer:\s*(.*)', block)
        answer = answer_match.group(1).strip() if answer_match else "Not provided"
        
        difficulty_match = re.search(r'Difficulty:\s*(.*)', block)
        difficulty = difficulty_match.group(1).strip() if difficulty_match else "Medium"
        
        questions.append({
            "question": question_text,
            "type": q_type,
            "options": options,
            "answer": answer,
            "difficulty": difficulty
        })
    
    return questions

def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate text to target language"""
    lang_codes = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
        "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
        "Malayalam": "ml", "Punjabi": "pa"
    }
    target_code = lang_codes.get(target_lang, "en")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source="auto", target=target_code)
            translated = translator.translate(text)
            if translated:
                return translated
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return text

def translate_test_suite(questions: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """Translate an entire test suite"""
    translated_questions = []
    
    for q in questions:
        if "error" in q:
            translated_questions.append(q)
            continue
        
        try:
            translated_q = {
                "question": translate_text(q["question"], target_lang=target_lang),
                "type": q["type"],
                "options": [translate_text(opt, target_lang=target_lang) for opt in q["options"]],
                "answer": translate_text(q["answer"], target_lang=target_lang),
                "difficulty": q["difficulty"]
            }
            translated_questions.append(translated_q)
        except Exception:
            translated_questions.append(q)
    
    return translated_questions

def format_test_suite(questions: List[Dict[str, Any]]) -> str:
    """Format questions into a readable string"""
    test_content = "**Test Suite**:\n"
    
    for i, q in enumerate(questions, 1):
        if "error" in q:
            test_content += f"- Error: {q['error']}\n"
            continue
        
        test_content += f"- Question {i}: {q['question']}\n"
        test_content += f"  Type: {q['type']}\n"
        test_content += f"  Options: [{'; '.join(q['options'])}]\n" if q['options'] else "  Options: []\n"
        test_content += f"  Correct Answer: {q['answer']}\n"
        test_content += f"  Difficulty: {q['difficulty']}\n"
    
    return test_content

def detect_language(text: str) -> str:
    """Detect the language of given text"""
    try:
        detected = GoogleTranslator(source="auto", target="en").detect(text)
        lang_code = detected[1].split('-')[0] if isinstance(detected, tuple) else detected.split('-')[0]
        return lang_code
    except Exception:
        return "en"

def text_to_speech(text: str, lang: str = "en") -> str:
    """Convert text to speech audio"""
    lang_map = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
        "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
        "Malayalam": "ml", "Punjabi": "pa"
    }
    
    try:
        gtts_lang = lang_map.get(lang, "en")
        translated_text = translate_text(text, target_lang=lang)
        tts = gTTS(text=translated_text, lang=gtts_lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return base64.b64encode(audio_bytes.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Text-to-speech failed: {str(e)}")
        return ""

def play_audio(audio_base64: str):
    """Play audio from base64 string"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")

def generate_lq_chart(student_data: Dict[str, float]) -> str:
    """Generate learning quotient chart"""
    if not student_data or not all(key in student_data for key in ["accuracy", "time_taken", "hints_used"]):
        return "Error: Invalid student data"
    
    metrics = ["Accuracy", "Time Taken (min)", "Hints Used"]
    values = [student_data["accuracy"], student_data["time_taken"], student_data["hints_used"]]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics, values, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_title("Learning Quotient (LQ) Metrics")
    ax.set_ylabel("Value")
    plt.tight_layout()
    
    chart_path = "temp_chart.png"
    fig.savefig(chart_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    
    return chart_path

def generate_pdf_report(questions: List[Dict[str, Any]], analysis: str, chart_path: str) -> str:
    """Generate PDF report using standard fonts only"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Use standard FPDF fonts (Helvetica, Times, Courier)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Sahayak Assessment Report", ln=1, align="C")
        
        # Questions section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Generated Questions", ln=1)
        pdf.set_font("Helvetica", "", 10)
        
        for q in questions:
            if "error" in q:
                pdf.multi_cell(0, 8, q["error"])
                continue
            
            # Handle non-ASCII characters by replacing unsupported ones
            question_text = q['question'].encode('ascii', 'replace').decode('ascii')
            pdf.multi_cell(0, 8, f"Question: {question_text}")
            
            pdf.cell(0, 8, f"Type: {q['type']}", ln=1)
            
            if q['options']:
                options_text = "; ".join(
                    opt.encode('ascii', 'replace').decode('ascii') 
                    for opt in q['options']
                )
                pdf.multi_cell(0, 8, f"Options: {options_text}")
            else:
                pdf.cell(0, 8, "Options: None", ln=1)
            
            answer_text = q['answer'].encode('ascii', 'replace').decode('ascii')
            pdf.cell(0, 8, f"Correct Answer: {answer_text}", ln=1)
            pdf.cell(0, 8, f"Difficulty: {q['difficulty']}", ln=1)
            pdf.ln(5)
        
        # Analysis section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Analysis", ln=1)
        pdf.set_font("Helvetica", "", 10)
        
        analysis_text = analysis.encode('ascii', 'replace').decode('ascii') if analysis else "No analysis provided"
        pdf.multi_cell(0, 8, analysis_text)
        pdf.ln(10)
        
        # Add chart if available
        if chart_path and os.path.exists(chart_path):
            pdf.image(chart_path, w=180)
        
        # Footer
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} IST", 0, 0, "C")
        
        pdf_path = "sahayak_report.pdf"
        pdf.output(pdf_path)
        
        return pdf_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return ""

# ------------------------ Custom CSS ------------------------ #
st.markdown("""
    <style>
        .main {
            background-color: #f5f9ff;
        }
        .stApp {
            background-image: linear-gradient(to bottom, #ffffff, #f0f7ff);
        }
        .sidebar .sidebar-content {
            background-color: #e3f2fd;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #e3f2fd;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #3e8e41;
            transform: scale(1.02);
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 8px;
            padding: 10px;
        }
        .stRadio>div {
            flex-direction: row;
            gap: 1rem;
        }
        .stRadio>div>label {
            margin-bottom: 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 2rem;
            background-color: #e3f2fd;
            border-radius: 0;
            margin-right: 0 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2196F3;
            color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stProgress>div>div>div>div {
            background-color: #2196F3;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stExpander {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .reportview-container .markdown-text-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .css-1aumxhk {
            background-color: #e3f2fd;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------ Main UI ------------------------ #
# Header with logo and title
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
        <img src="https://img.icons8.com/color/48/000000/book.png" width="48">
        <div>
            <h1 style="margin: 0; color: #2196F3;">Sahayak: AI Education Platform</h1>
            <p style="margin: 0; color: #555; font-size: 1.1rem;">Empowering teachers in low-resource classrooms</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Status bar
st.markdown(f"""
    <div style="background-color: #e3f2fd; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>📅 Current Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} IST
            </div>
            <div>
                <strong>🏫 Class Level:</strong> {st.session_state.class_level} | 
                <strong>🌐 Language:</strong> {st.session_state.language}
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Test Creation", 
    "💡 Hint Generator", 
    "✅ Grading Assistant", 
    "📊 Learning Analytics"
])

# ------------------------ Test Creation Tab ------------------------ #
with tab1:
    with st.container():
        st.markdown("""
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #2196F3;">Create Custom Assessments</h3>
                <p style="margin: 0.5rem 0 0; color: #555;">Generate tests from various input sources</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            input_type = st.radio(
                "Select Input Type",
                ["Textbook Photo", "Text Description", "URL"],
                key="test_creation_input_type",
                horizontal=True
            )
            
            # Initialize input_data as None
            input_data = None
            
            if input_type == "Textbook Photo":
                img_file = st.file_uploader(
                    "Upload textbook page",
                    type=["jpg", "jpeg", "png"],
                    key="test_creation_image",
                    help="Take a clear photo of the textbook page or worksheet"
                )
                if img_file:
                    input_data = Image.open(img_file)
                    with st.expander("Preview Uploaded Image", expanded=True):
                        st.image(input_data, use_column_width=True)
            elif input_type == "Text Description":
                input_data = st.text_area(
                    "Describe the topic or paste content",
                    placeholder="E.g. Fractions for Class 5\n- What is 1/2 + 1/4?\n- Explain numerator and denominator",
                    key="test_creation_text",
                    height=150
                )
            else:  # URL
                input_data = st.text_input(
                    "Enter educational content URL",
                    placeholder="E.g. https://example.com/science-lesson",
                    key="test_creation_url"
                )
        
        with col2:
            context = st.text_area(
                "Additional Context (Optional)",
                placeholder="E.g. Focus on word problems\nUse mango examples\nInclude 2 diagram questions",
                key="test_creation_context",
                height=150
            )
            
            col21, col22 = st.columns([1, 1])
            with col21:
                num_questions = st.number_input(
                    "Number of Questions",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="test_creation_num_questions"
                )
            with col22:
                translate_to = st.selectbox(
                    "Translate To",
                    ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
                    key="translate_to",
                    index=0
                )
        
        # Check if input_data has been set (not None and not empty string)
        is_input_valid = False
        if input_data is not None:
            if isinstance(input_data, str):
                is_input_valid = bool(input_data.strip())
            else:  # For image objects
                is_input_valid = True
        
        if st.button(
            "✨ Generate Assessment",
            disabled=not is_input_valid,
            key="test_creation_generate",
            help="Generate customized test based on your inputs"
        ):
            with st.spinner("🧠 Generating your assessment... This may take a moment"):
                try:
                    # Process input based on type
                    if input_type == "Textbook Photo":
                        content = process_image(input_data)
                    elif input_type == "URL":
                        content = process_url(input_data)
                    else:
                        content = input_data

                    if "Error" in content:
                        st.error(content)
                        st.stop()

            
                    # Generate test content
                    prompt = get_test_creation_prompt(content, context, st.session_state.class_level, num_questions)
                    test_content = query_langchain(prompt)
                    questions = extract_questions_and_answers(test_content)

                    # Retry if parsing fails
                    if questions and "error" in questions[0]:
                        st.warning("Parsing failed. Retrying...")
                        simplified_prompt = get_test_creation_retry_prompt(content, context, st.session_state.class_level, num_questions)
                        test_content = query_langchain(simplified_prompt, retry=True)
                        questions = extract_questions_and_answers(test_content)

                    if questions and "error" in questions[0]:
                        st.error(f"Failed to parse questions: {questions[0]['error']}")
                        st.stop()

                    # Prepare original and translated versions
                    original_questions = questions
                    original_test_content = format_test_suite(original_questions)
                    translated_questions = translate_test_suite(questions, translate_to)
                    translated_test_content = format_test_suite(translated_questions)

                    # Display results
                    st.success("✅ Assessment generated successfully!")
                    
                    with st.expander("📄 Original Test (English)", expanded=True):
                        st.markdown(original_test_content)
                    
                    with st.expander(f"🌍 Translated Test ({translate_to})", expanded=False):
                        st.markdown(translated_test_content)
                    
                    # Audio options
                    if st.checkbox("🔊 Enable audio for questions", key="enable_audio"):
                        for i, q in enumerate(translated_questions, 1):
                            if "error" not in q:
                                audio_base64 = text_to_speech(f"Question {i}: {q['question']}", translate_to)
                                if audio_base64:
                                    st.markdown(f"**Question {i} Audio:**")
                                    play_audio(audio_base64)
                    
                    # Store results
                    st.session_state.last_results = {
                        "type": "test",
                        "input_type": input_type,
                        "input_data": content,
                        "context": context,
                        "questions": translated_questions,
                        "original_questions": original_questions,
                        "test_content": translated_test_content,
                        "original_test_content": original_test_content,
                        "translate_to": translate_to,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.test_history.append(st.session_state.last_results)
                    
                    # Generate PDF
                    pdf_path = generate_pdf_report(
                        translated_questions,
                        translated_test_content,
                        None
                    )
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 Download Test as PDF",
                                f,
                                file_name=f"sahayak_test_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                key="test_download_pdf"
                            )
                        os.remove(pdf_path)

                except Exception as e:
                    st.error(f"❌ Test generation failed: {str(e)}")
                    st.error(traceback.format_exc())

# ------------------------ Hint Generator Tab ------------------------ #
with tab2:
    with st.container():
        st.markdown("""
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #2196F3;">Personalized Hint Generator</h3>
                <p style="margin: 0.5rem 0 0; color: #555;">Create scaffolded hints for struggling students</p>
            </div>
        """, unsafe_allow_html=True)
        
        question = st.text_area(
            "Enter the Question",
            placeholder="E.g. Solve for x: 2x + 5 = 15",
            key="hint_generator_question",
            height=100
        )
        
        student_response = st.text_area(
            "Student's Attempt (Optional)",
            placeholder="E.g. x = 10",
            key="hint_generator_response",
            height=100,
            help="Provide the student's answer (if available) for more targeted hints"
        )
        
        hint_level = st.slider(
            "Hint Specificity Level",
            min_value=1,
            max_value=3,
            value=2,
            key="hint_level",
            help="1: General guidance, 3: Specific step-by-step help"
        )
        
        if st.button(
            "💡 Generate Hint",
            disabled=not question,
            key="hint_generator_generate",
            help="Create a scaffolded hint based on the question"
        ):
            with st.spinner("🤔 Crafting the perfect hint..."):
                try:
                    prompt = get_hint_generator_prompt(question, student_response, hint_level)
                    hint = query_langchain(prompt)
                    
                    if st.session_state.language != "English":
                        hint = translate_text(hint, target_lang=st.session_state.language)
                    
                    st.success("✅ Hint generated successfully!")
                    
                    # Display hint in styled box
                    st.markdown(f"""
                        <div style="
                            background-color: #e8f5e9;
                            border-left: 4px solid #4CAF50;
                            padding: 1rem;
                            border-radius: 0 8px 8px 0;
                            margin: 1rem 0;
                        ">
                            <h4 style="margin: 0 0 0.5rem; color: #2e7d32;">Generated Hint:</h4>
                            <p style="margin: 0;">{hint}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add audio option
                    audio_base64 = text_to_speech(hint, st.session_state.language)
                    if audio_base64:
                        st.markdown("🔊 Listen to hint:")
                        play_audio(audio_base64)
                    
                    # Store results
                    st.session_state.test_history.append({
                        "type": "hint",
                        "question": question,
                        "student_response": student_response,
                        "hint": hint,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                except Exception as e:
                    st.error(f"❌ Hint generation failed: {str(e)}")

# ------------------------ Grading Tab ------------------------ #
with tab3:
    with st.container():
        st.markdown("""
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #2196F3;">Automated Grading Assistant</h3>
                <p style="margin: 0.5rem 0 0; color: #555;">Evaluate student responses with AI feedback</p>
            </div>
        """, unsafe_allow_html=True)
        
        response_type = st.radio(
            "Response Format",
            ["Text Response", "Handwritten Response"],
            key="grading_response_type",
            horizontal=True
        )
        
        if response_type == "Text Response":
            student_response = st.text_area(
                "Student's Answer",
                placeholder="E.g. The capital of India is Delhi",
                key="grading_text_response",
                height=150
            )
        else:
            img_file = st.file_uploader(
                "Upload Handwritten Answer",
                type=["jpg", "jpeg", "png"],
                key="grading_image",
                help="Upload a clear photo of the student's handwritten work"
            )
            if img_file:
                image = Image.open(img_file)
                with st.expander("Preview Handwritten Answer", expanded=True):
                    st.image(image, width=300)
        
        correct_answer = st.text_input(
            "Correct Answer",
            placeholder="E.g. Delhi",
            key="grading_correct_answer"
        )
        
        rubric = st.text_area(
            "Grading Rubric (Optional)",
            placeholder="E.g. 2 points for correct answer, 1 point for partial answer",
            key="grading_rubric",
            height=100
        )
        
        if st.button(
            "📝 Grade Response",
            disabled=not (correct_answer and (student_response or img_file)),
            key="grading_generate_text"
        ):
            with st.spinner("🔍 Analyzing response..."):
                try:
                    # Process response based on type
                    if response_type == "Handwritten Response":
                        description = process_image(image)
                        if "Error" in description:
                            st.error(description)
                            st.stop()
                        detected_lang = detect_language(description)
                        st.write(f"Detected language: {detected_lang}")
                        translated_response = translate_text(description, target_lang="en") if detected_lang != "en" else description
                        st.write(f"Translated to English: {translated_response}")
                        responses = [{
                            "response": translated_response,
                            "original_response": description,
                            "correct_answer": correct_answer,
                            "rubric": rubric
                        }]
                    else:
                        detected_lang = detect_language(student_response)
                        st.write(f"Detected language: {detected_lang}")
                        translated_response = translate_text(student_response, target_lang="en") if detected_lang != "en" else student_response
                        st.write(f"Translated to English: {translated_response}")
                        responses = [{
                            "response": translated_response,
                            "original_response": student_response,
                            "correct_answer": correct_answer,
                            "rubric": rubric
                        }]
                    
                    # Get grading result
                    prompt = get_grading_prompt(responses[0]["response"], responses[0]["correct_answer"], responses[0]["rubric"])
                    grading_result = query_langchain(prompt)
                    
                    # Translate feedback if needed
                    if st.session_state.language != "English":
                        feedback_match = re.search(r'\*\*Feedback\*\*: (.*)', grading_result, re.DOTALL)
                        if feedback_match:
                            feedback = feedback_match.group(1)
                            translated_feedback = translate_text(feedback, target_lang=st.session_state.language)
                            grading_result = grading_result.replace(feedback, translated_feedback)
                    
                    st.success("✅ Grading complete!")
                    
                    # Display results in styled box
                    st.markdown(f"""
                        <div style="
                            background-color: #fff8e1;
                            border-left: 4px solid #ffc107;
                            padding: 1rem;
                            border-radius: 0 8px 8px 0;
                            margin: 1rem 0;
                        ">
                            <h4 style="margin: 0 0 0.5rem; color: #ff8f00;">Grading Results:</h4>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Student Response:</strong> {responses[0]['original_response']}
                            </div>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Correct Answer:</strong> {responses[0]['correct_answer']}
                            </div>
                            <div>{grading_result}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Store results
                    st.session_state.test_history.append({
                        "type": "grade",
                        "response": responses[0]["response"],
                        "original_response": responses[0]["original_response"],
                        "correct_answer": responses[0]["correct_answer"],
                        "rubric": responses[0]["rubric"],
                        "grading_result": grading_result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                except Exception as e:
                    st.error(f"❌ Grading failed: {str(e)}")

# ------------------------ Analytics Tab ------------------------ #
with tab4:
    with st.container():
        st.markdown("""
            <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #2196F3;">Learning Analytics Dashboard</h3>
                <p style="margin: 0.5rem 0 0; color: #555;">Track student progress and performance</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            student_id = st.text_input(
                "Student ID",
                placeholder="E.g. STU001",
                key="analytics_student_id"
            )
        with col2:
            topic = st.text_input(
                "Topic/Subject",
                placeholder="E.g. Fractions",
                key="analytics_topic"
            )
        
        time_range = st.selectbox(
            "Time Period",
            ["Last Week", "Last Month", "Last Quarter", "All Time"],
            key="analytics_time_range"
        )
        
        if st.button(
            "📈 Generate Insights",
            disabled=not (student_id and topic),
            key="analytics_generate"
        ):
            with st.spinner("📊 Analyzing learning patterns..."):
                try:
                    prompt = get_analytics_prompt(student_id, topic, time_range)
                    insights = query_langchain(prompt)
                    translated_insights = translate_text(insights, target_lang=st.session_state.language) if st.session_state.language != "English" else insights
                    
                    st.success("✅ Analytics generated!")
                    
                    # Display insights in styled box
                    st.markdown(f"""
                        <div style="
                            background-color: #f3e5f5;
                            border-left: 4px solid #9c27b0;
                            padding: 1rem;
                            border-radius: 0 8px 8px 0;
                            margin: 1rem 0;
                        ">
                            <h4 style="margin: 0 0 0.5rem; color: #7b1fa2;">Learning Insights for {student_id}:</h4>
                            <div>{translated_insights}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate and display LQ chart
                    student_data = {"accuracy": 85, "time_taken": 30, "hints_used": 2}
                    chart_path = generate_lq_chart(student_data)
                    if chart_path and os.path.exists(chart_path):
                        st.image(chart_path, use_column_width=True)
                    
                    # Store results
                    st.session_state.test_history.append({
                        "type": "insights",
                        "student_id": student_id,
                        "topic": topic,
                        "insights": translated_insights,
                        "chart_path": chart_path,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                    # Generate PDF report if this follows a test generation
                    if st.session_state.last_results.get("type") == "test":
                        pdf_path = generate_pdf_report(
                            st.session_state.last_results.get("questions", []),
                            st.session_state.last_results.get("test_content", ""),
                            chart_path
                        )
                        if pdf_path and os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "📄 Download Full Report",
                                    f,
                                    file_name=f"learning_report_{student_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    key="analytics_download_report"
                                )
                            os.remove(pdf_path)
                            if os.path.exists(chart_path):
                                os.remove(chart_path)
                    
                except Exception as e:
                    st.error(f"❌ Analytics failed: {str(e)}")

# ------------------------ Sidebar ------------------------ #
with st.sidebar:
    st.markdown("""
        <div style="
            background-color: #2196F3;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        ">
            <h2 style="margin: 0; color: white;">Teacher Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Class Settings section - no expander needed since it's already in sidebar
    st.subheader("⚙️ Class Settings")
    class_level = st.selectbox(
        "Grade Level",
        ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8"],
        key="dashboard_class_level",
        index=4  # Default to Class 5
    )
    language = st.selectbox(
        "Feedback Language",
        ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
        key="dashboard_language"
    )
    st.session_state.class_level = class_level
    st.session_state.language = language
    
    # Recent Activities section - using a single expander
    st.subheader("📚 Recent Activities")
    if not st.session_state.test_history:
        st.info("No recent activities yet")
    else:
        # Create a selectbox to choose which activity to view details for
        activity_options = [
            f"{entry['timestamp']} - {entry['type'].title()}" 
            for entry in reversed(st.session_state.test_history[-5:])
        ]
        selected_activity = st.selectbox(
            "Select activity to view details",
            activity_options,
            key="activity_selector"
        )
        
        # Find the selected activity
        selected_entry = next(
            entry for entry in reversed(st.session_state.test_history[-5:])
            if f"{entry['timestamp']} - {entry['type'].title()}" == selected_activity
        )
        
        # Display details of the selected activity
        if selected_entry['type'] == "test":
            st.markdown(f"**Subject:** {selected_entry.get('context', 'N/A')}")
            st.markdown(f"**Questions:** {len(selected_entry.get('questions', []))}")
            st.markdown(f"**Language:** {selected_entry.get('translate_to', 'English')}")
        elif selected_entry['type'] == "hint":
            st.markdown(f"**Question:** {selected_entry.get('question', 'N/A')}")
            st.markdown(f"**Hint:** {selected_entry.get('hint', 'N/A')}")
        elif selected_entry['type'] == "grade":
            st.markdown(selected_entry.get("grading_result", "N/A"))
        elif selected_entry['type'] == "insights":
            st.markdown(f"**Student:** {selected_entry.get('student_id', 'N/A')}")
            st.markdown(f"**Topic:** {selected_entry.get('topic', 'N/A')}")
            st.markdown(selected_entry.get("insights", "N/A"))
            if selected_entry.get("chart_path") and os.path.exists(selected_entry["chart_path"]):
                st.image(selected_entry["chart_path"])
    
    if st.button(
        "🗑️ Clear All History",
        key="dashboard_clear_history",
        help="Remove all stored activities"
    ):
        for entry in st.session_state.test_history:
            if entry.get("chart_path") and os.path.exists(entry["chart_path"]):
                os.remove(entry["chart_path"])
        st.session_state.test_history.clear()
        st.session_state.last_results = {}
        st.success("History cleared successfully!")
        time.sleep(1)
        st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>📱 Sahayak Education Platform v1.0</p>
            <p>© 2025 AI for Education Initiative</p>
        </div>
    """, unsafe_allow_html=True)