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
from prompts import (
    get_test_creation_prompt,
    get_test_creation_retry_prompt,
    get_hint_generator_prompt,
    get_grading_prompt,
    get_analytics_prompt
)

# ------------------------ Setup ------------------------ #
st.set_page_config(page_title="📚 Sahayak: AI Education Platform", layout="wide")
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()
translator = GoogleTranslator(source="auto", target="en")

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    models = {}
    try:
        models['llm'] = ChatGroq(model_name="gemma2-9b-it", api_key=groq_api_key)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        models['llm'] = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=dtype
        ).to(device)
        models['blip_model'].eval()
    except Exception as e:
        st.error(f"Failed to load image processor: {e}")
        models['processor'] = None
        models['blip_model'] = None
    return models

models = load_models()

# ------------------------ Session State ------------------------ #
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
    if not models['llm']:
        return "Error: LLM service unavailable"
    try:
        response = models['llm'].invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"

def process_image(image: Image.Image, use_ocr: bool = True) -> str:
    if use_ocr:
        try:
            text = pytesseract.image_to_string(image, lang="eng+hin+tam+ben+tel+mar+guj+kan+mal+pan")
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
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=10000)
            page.wait_for_load_state("domcontentloaded")
            content = page.content()
            text = page.evaluate(
                """() => {
                    // Remove scripts, styles, and other non-content elements
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
    questions = []
    pattern = r'Question \d+: (.*?)\n(?:Type: (.*?)\n)?Options: (.*?)\nCorrect Answer: (.*?)\nDifficulty: (.*?)(?:\n|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.UNICODE)
    for match in matches:
        question, q_type, options, answer, difficulty = match
        questions.append({
            "question": question.strip(),
            "type": q_type.strip() if q_type else "MCQ",
            "options": [opt.strip() for opt in options.split(";") if opt.strip()] if options.strip() else [],
            "answer": answer.strip(),
            "difficulty": difficulty.strip() if difficulty else "Medium"
        })
    if not matches:
        fallback_pattern = r'Question \d+: (.*?)(?:\n|$)(?:Type: (.*?)(?:\n|$))?(?:Options: (.*?)(?:\n|$))?(?:Correct Answer: (.*?)(?:\n|$))?(?:Difficulty: (.*?)(?:\n|$))?'
        fallback_matches = re.findall(fallback_pattern, text, re.DOTALL | re.MULTILINE | re.UNICODE)
        for match in fallback_matches:
            question, q_type, options, answer, difficulty = match
            if question.strip():
                questions.append({
                    "question": question.strip(),
                    "type": q_type.strip() if q_type else "MCQ",
                    "options": [opt.strip() for opt in options.split(";") if opt.strip()] if options.strip() else [],
                    "answer": answer.strip() if answer else "Not provided",
                    "difficulty": difficulty.strip() if difficulty else "Medium"
                })
    if not questions:
        question_blocks = text.split("- Question")
        for block in question_blocks[1:]:
            block = block.strip()
            if not block:
                continue
            q_match = re.match(r'^\d+: (.*?)(?=\n|$)', block, re.DOTALL | re.UNICODE)
            if not q_match:
                continue
            question = q_match.group(1).strip()
            type_match = re.search(r'Type: (.*?)(?=\n|$)', block, re.DOTALL | re.UNICODE)
            options_match = re.search(r'Options: (.*?)(?=\n|$)', block, re.DOTALL | re.UNICODE)
            answer_match = re.search(r'Correct Answer: (.*?)(?=\n|$)', block, re.DOTALL | re.UNICODE)
            difficulty_match = re.search(r'Difficulty: (.*?)(?=\n|$)', block, re.DOTALL | re.UNICODE)
            questions.append({
                "question": question,
                "type": type_match.group(1).strip() if type_match else "MCQ",
                "options": [opt.strip() for opt in options_match.group(1).split(";") if opt.strip()] if options_match and options_match.group(1).strip() else [],
                "answer": answer_match.group(1).strip() if answer_match else "Not provided",
                "difficulty": difficulty_match.group(1).strip() if difficulty_match else "Medium"
            })
    return questions if questions else [{"error": "No questions parsed from output"}]

def translate_text(text: str, target_lang: str = "en") -> str:
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
            st.warning(f"Translation returned None for text '{text[:50]}...' on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            st.error(f"Translation failed for text '{text[:50]}...' on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
    st.error(f"Translation failed after {max_retries} attempts. Using original text.")
    return text

def translate_test_suite(questions: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    lang_codes = {
        "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
        "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
        "Malayalam": "ml", "Punjabi": "pa"
    }
    target_code = lang_codes.get(target_lang, "en")
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
        except Exception as e:
            st.warning(f"Translation failed for question '{q['question'][:50]}...': {str(e)}")
            translated_questions.append(q)
    return translated_questions

def format_test_suite(questions: List[Dict[str, Any]]) -> str:
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
    try:
        detected = GoogleTranslator(source="auto", target="en").detect(text)
        lang_code = detected[1].split('-')[0] if isinstance(detected, tuple) else detected.split('-')[0]
        return lang_code
    except Exception as e:
        st.error(f"Language detection failed: {str(e)}")
        return "en"

def text_to_speech(text: str, lang: str = "en") -> str:
    try:
        lang_map = {
            "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
            "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
            "Malayalam": "ml", "Punjabi": "pa"
        }
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
    try:
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")

def generate_lq_chart(student_data: Dict[str, float]) -> str:
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
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Sahayak Assessment Report", ln=1, align="C")
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Generated Questions", ln=1)
        pdf.set_font("Arial", "", 10)
        for q in questions:
            if "error" in q:
                pdf.multi_cell(0, 8, q["error"])
                continue
            pdf.multi_cell(0, 8, f"Question: {q['question']}")
            pdf.cell(0, 8, f"Type: {q['type']}", ln=1)
            pdf.cell(0, 8, f"Options: {'; '.join(q['options'])}" if q['options'] else "None", ln=1)
            pdf.cell(0, 8, f"Correct Answer: {q['answer']}", ln=1)
            pdf.cell(0, 8, f"Difficulty: {q['difficulty']}", ln=1)
            pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analysis", ln=1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 8, analysis or "No analysis provided")
        pdf.ln(10)
        if chart_path and os.path.exists(chart_path):
            pdf.image(chart_path, w=180)
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} IST", 0, 0, "C")
        pdf_path = "sahayak_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return ""

# ------------------------ Streamlit UI ------------------------ #
st.title("📚 Sahayak: AI Education Platform")
st.caption("Empowering teachers in low-resource classrooms")
st.markdown(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Test Creation", "💡 Hint Generator", "✅ Grading", "📊 Analytics"])

with tab1:
    st.subheader("Create Assessments")
    input_type = st.radio("Input Type", ["Textbook Photo", "Text Description", "URL"], key="test_creation_input_type")
    input_data = None
    if input_type == "Textbook Photo":
        img_file = st.file_uploader("Upload textbook page", type=["jpg", "jpeg", "png"], key="test_creation_image")
        if img_file:
            input_data = Image.open(img_file)
            st.image(input_data, use_column_width=True)
    elif input_type == "Text Description":
        input_data = st.text_area("Describe topic", placeholder="E.g. Fractions for Class 5", key="test_creation_text")
    else:  # URL
        input_data = st.text_input("Enter URL", placeholder="E.g. https://example.com/science-lesson", key="test_creation_url")
    
    context = st.text_area("Context", placeholder="E.g. Class 5, use mango examples", key="test_creation_context")
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5, key="test_creation_num_questions")
    
    translate_to = st.selectbox("Translate Questions To", ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"], key="translate_to")
    
    if st.button("Generate Test", disabled=not input_data, key="test_creation_generate"):
        with st.spinner("Generating test..."):
            try:
                if input_type == "Textbook Photo":
                    content = process_image(input_data)
                elif input_type == "URL":
                    content = process_url(input_data)
                else:
                    content = input_data
                if "Error" in content:
                    st.error(content)
                    st.stop()
                
                prompt = get_test_creation_prompt(content, context, st.session_state.class_level, num_questions)
                test_content = query_langchain(prompt)
                questions = extract_questions_and_answers(test_content)
                
                if questions and "error" in questions[0]:
                    st.warning("Parsing failed. Retrying...")
                    simplified_prompt = get_test_creation_retry_prompt(content, context, st.session_state.class_level, num_questions)
                    test_content = query_langchain(simplified_prompt, retry=True)
                    questions = extract_questions_and_answers(test_content)
                
                if questions and "error" in questions[0]:
                    st.error(f"Failed to parse questions: {questions[0]['error']}")
                    st.stop()
                
                original_questions = questions
                original_test_content = format_test_suite(original_questions)
                
                translated_questions = translate_test_suite(questions, translate_to)
                translated_test_content = format_test_suite(translated_questions)
                
                st.subheader("Original Test (English)")
                st.markdown(original_test_content)
                st.subheader(f"Translated Test ({translate_to})")
                st.markdown(translated_test_content)
                
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
                
            except Exception as e:
                st.error(f"Test generation failed: {str(e)}")

with tab2:
    st.subheader("Generate Hints")
    question = st.text_area("Enter Question", placeholder="E.g. What is the capital of India?", key="hint_generator_question")
    student_response = st.text_area("Student Response (Optional)", placeholder="E.g. Mumbai", key="hint_generator_response")
    
    if st.button("Generate Hint", disabled=not question, key="hint_generator_generate"):
        with st.spinner("Generating hint..."):
            try:
                prompt = get_hint_generator_prompt(question, student_response)
                hint = query_langchain(prompt)
                if st.session_state.language != "English":
                    hint = translate_text(hint, target_lang=st.session_state.language)
                st.markdown(hint)
                st.session_state.test_history.append({
                    "type": "hint",
                    "question": question,
                    "student_response": student_response,
                    "hint": hint,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
            except Exception as e:
                st.error(f"Hint generation failed: {str(e)}")

with tab3:
    st.subheader("Grade Responses")
    response_type = st.radio("Response Type", ["Text", "Handwritten Image"], key="grading_response_type")
    responses = []
    
    if response_type == "Text":
        student_response = st.text_area("Enter Student Response", placeholder="E.g. The capital of India is Delhi", key="grading_text_response")
        correct_answer = st.text_input("Correct Answer", placeholder="E.g. Delhi", key="grading_correct_answer")
        rubric = st.text_area("Grading Rubric", placeholder="E.g. 2 points for correct answer", key="grading_rubric")
        
        if student_response:
            detected_lang = detect_language(student_response)
            st.write(f"Detected language: {detected_lang}")
            translated_response = translate_text(student_response, target_lang="en") if detected_lang != "en" else student_response
            st.write(f"Translated to English: {translated_response}")
        
        if st.button("Grade Response", disabled=not (student_response and correct_answer and rubric), key="grading_generate_text"):
            responses.append({"response": translated_response, "original_response": student_response, "correct_answer": correct_answer, "rubric": rubric})
    else:
        img_file = st.file_uploader("Upload Handwritten Response", type=["jpg", "jpeg", "png"], key="grading_image")
        correct_answer = st.text_input("Correct Answer", key="grading_image_correct_answer")
        rubric = st.text_area("Grading Rubric", key="grading_image_rubric")
        if img_file and st.button("Grade Response", disabled=not (correct_answer and rubric), key="grading_generate_image"):
            image = Image.open(img_file)
            st.image(image, width=300)
            description = process_image(image)
            if "Error" in description:
                st.error(description)
                st.stop()
            detected_lang = detect_language(description)
            st.write(f"Detected language: {detected_lang}")
            translated_response = translate_text(description, target_lang="en") if detected_lang != "en" else description
            st.write(f"Translated to English: {translated_response}")
            responses.append({"response": translated_response, "original_response": description, "correct_answer": correct_answer, "rubric": rubric})
    
    if responses:
        with st.spinner("Grading..."):
            try:
                prompt = get_grading_prompt(responses[0]["response"], responses[0]["correct_answer"], responses[0]["rubric"])
                grading_result = query_langchain(prompt)
                if st.session_state.language != "English":
                    feedback_match = re.search(r'\*\*Feedback\*\*: (.*)', grading_result, re.DOTALL)
                    if feedback_match:
                        feedback = feedback_match.group(1)
                        translated_feedback = translate_text(feedback, target_lang=st.session_state.language)
                        grading_result = grading_result.replace(feedback, translated_feedback)
                st.markdown(grading_result)
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
                st.error(f"Grading failed: {str(e)}")

with tab4:
    st.subheader("Learning Analytics")
    student_id = st.text_input("Student ID", placeholder="E.g. STU001", key="analytics_student_id")
    topic = st.text_input("Topic", placeholder="E.g. Fractions", key="analytics_topic")
    
    if st.button("Generate Insights", disabled=not (student_id and topic), key="analytics_generate"):
        with st.spinner("Generating insights..."):
            try:
                prompt = get_analytics_prompt(student_id, topic)
                insights = query_langchain(prompt)
                translated_insights = translate_text(insights, target_lang=st.session_state.language) if st.session_state.language != "English" else insights
                st.markdown(translated_insights)
                student_data = {"accuracy": 85, "time_taken": 30, "hints_used": 2}
                chart_path = generate_lq_chart(student_data)
                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path)
                st.session_state.test_history.append({
                    "type": "insights",
                    "student_id": student_id,
                    "topic": topic,
                    "insights": translated_insights,
                    "chart_path": chart_path,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
            except Exception as e:
                st.error(f"Analytics failed: {str(e)}")
    
    if st.session_state.last_results.get("type") == "test":
        if st.button("📄 Generate Test Report", key="analytics_generate_report"):
            try:
                pdf_path = generate_pdf_report(
                    st.session_state.last_results.get("questions", []),
                    st.session_state.last_results.get("test_content", ""),
                    st.session_state.test_history[-1].get("chart_path") if st.session_state.test_history else None
                )
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download Report", f, file_name="sahayak_report.pdf", mime="application/pdf", key="analytics_download_report")
                    os.remove(pdf_path)
                    if st.session_state.test_history and st.session_state.test_history[-1].get("chart_path") and os.path.exists(st.session_state.test_history[-1]["chart_path"]):
                        os.remove(st.session_state.test_history[-1]["chart_path"])
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

with st.sidebar:
    st.header("Teacher Dashboard")
    st.subheader("Class Settings")
    class_level = st.selectbox("Class Level", [f"Class {i}" for i in range(1, 13)], key="dashboard_class_level")
    language = st.selectbox("Feedback Language", ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"], key="dashboard_language")
    st.session_state.class_level = class_level
    st.session_state.language = language
    
    st.subheader("Test History")
    for entry in reversed(st.session_state.test_history):
        with st.expander(f"{entry['timestamp']} - {entry['type'].title()}"):
            if entry['type'] == "test":
                st.markdown("**Original (English)**:")
                st.markdown(entry["original_test_content"])
                st.markdown(f"**Translated ({entry.get('translate_to', 'Unknown')})**:")
                st.markdown(entry["test_content"])
            elif entry['type'] == "hint":
                st.markdown(f"**Question**: {entry['question']}\n**Hint**: {entry['hint']}")
            elif entry['type'] == "grade":
                st.markdown(entry["grading_result"])
                st.write(f"Original Response: {entry['original_response']}")
            elif entry['type'] == "insights":
                st.markdown(entry["insights"])
                if entry.get("chart_path") and os.path.exists(entry["chart_path"]):
                    st.image(entry["chart_path"])
    
    if st.button("Clear History", key="dashboard_clear_history"):
        for entry in st.session_state.test_history:
            if entry.get("chart_path") and os.path.exists(entry["chart_path"]):
                os.remove(entry["chart_path"])
        st.session_state.test_history.clear()
        st.session_state.last_results = {}
        st.rerun()

