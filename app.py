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

# ------------------------ Setup ------------------------ #
st.set_page_config(page_title="📚 Sahayak: AI Education Platform", layout="wide")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it up.")
    st.stop()

# Initialize Google Translator
translator = GoogleTranslator(source="auto", target="en")

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    """Initialize AI models without caching"""
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
            "Salesforce/blip-image-captioning-base", torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        models['blip_model'].eval()
    except Exception as e:
        st.error(f"Failed to load image processor: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        models['processor'] = None
        models['blip_model'] = None

    return models

models = load_models()

# ------------------------ Session State Initialization ------------------------ #
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
    """Query Groq LLM with error handling and optional retry"""
    if not models['llm']:
        return "Error: LLM service unavailable"
    try:
        response = models['llm'].invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"

def process_image(image: Image.Image, use_ocr: bool = True) -> str:
    """Extract text from images using OCR or describe content using BLIP"""
    if use_ocr:
        try:
            text = pytesseract.image_to_string(image, lang="eng+hin+tam+ben+tel+mar+guj+kan+mal+pan")
            if text.strip():
                return text.strip()
            st.warning("OCR extracted no text. Falling back to BLIP captioning.")
        except Exception as e:
            st.warning(f"OCR failed: {str(e)}. Falling back to BLIP captioning.")
    
    if not models['processor'] or not models['blip_model']:
        return "Error: Image processing unavailable"
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        device = next(models['blip_model'].parameters()).device
        inputs = models['processor'](images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = models['blip_model'].generate(**inputs, max_new_tokens=100)
        description = models['processor'].decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not description:
            return "Error: No content extracted from image"
        return description
    except Exception as e:
        return f"Error: Image processing failed - {str(e)}"

def extract_questions_and_answers(text: str) -> List[Dict[str, Any]]:
    """Extract questions, answers, and metadata from LLM output with robust parsing"""
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
    
    if not questions:
        return [{"error": "No questions parsed from output"}]
    return questions

def translate_test_suite(questions: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """Translate all questions, options, and answers in the test suite"""
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
        translated_q = {
            "question": translate_text(q["question"], target_lang=target_code),
            "type": q["type"],
            "options": [translate_text(opt, target_lang=target_code) for opt in q["options"]],
            "answer": translate_text(q["answer"], target_lang=target_code),
            "difficulty": q["difficulty"]
        }
        translated_questions.append(translated_q)
    
    return translated_questions

def generate_lq_chart(student_data: Dict[str, float]) -> str:
    """Generate Learning Quotient chart and save as PNG"""
    if not student_data or not all(key in student_data for key in ["accuracy", "time_taken", "hints_used"]):
        return "Error: Invalid or incomplete student data"
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
    """Generate PDF report with test details, analysis, and LQ chart"""
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

def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate text to target language with robust error handling"""
    try:
        # Normalize language codes
        lang_codes = {
            "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
            "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
            "Malayalam": "ml", "Punjabi": "pa"
        }
        target_code = lang_codes.get(target_lang, "en")
        translated = GoogleTranslator(source="auto", target=target_code).translate(text)
        return translated if translated else text
    except Exception as e:
        st.error(f"Translation failed for text '{text[:50]}...': {str(e)}")
        return text

def detect_language(text: str) -> str:
    """Detect the language of the input text"""
    try:
        detected = GoogleTranslator(source="auto", target="en").detect(text)
        # Normalize language code (e.g., 'hi-IN' -> 'hi')
        lang_code = detected[1].split('-')[0] if isinstance(detected, tuple) else detected.split('-')[0]
        return lang_code
    except Exception as e:
        st.error(f"Language detection failed: {str(e)}")
        return "en"

def text_to_speech(text: str, lang: str = "en") -> str:
    """Convert text to speech using gTTS and return base64 encoded audio"""
    try:
        lang_map = {
            "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
            "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
            "Malayalam": "ml", "Punjabi": "pa"
        }
        gtts_lang = lang_map.get(lang, "en")
        translated_text = translate_text(text, target_lang=gtts_lang)
        tts = gTTS(text=translated_text, lang=gtts_lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return base64.b64encode(audio_bytes.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Text-to-speech failed: {str(e)}")
        return ""

def play_audio(audio_base64: str):
    """Display audio player for base64 encoded audio"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")

# ------------------------ Streamlit UI ------------------------ #
st.title("📚 Sahayak: AI-Powered Education Platform")
st.caption("Empowering teachers in multi-grade, low-resource classrooms with automated assessments and analytics")
st.markdown(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST")

# Initialize tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Test Creation", "💡 Hint Generator", "✅ Grading", "📊 Analytics"])

# Test Creation Tab (TestCraft Agent)
with tab1:
    st.subheader("Create Assessments")
    input_type = st.radio(
        "Input Type",
        ["Textbook Photo", "Text Description", "Voice Instruction (Text Proxy)", "URL"],
        key="test_creation_input_type"
    )
    input_data = None
    
    if input_type == "Textbook Photo":
        img_file = st.file_uploader("Upload textbook page", type=["jpg", "jpeg", "png"], key="test_creation_image")
        if img_file:
            input_data = Image.open(img_file)
            st.image(input_data, use_column_width=True)
    elif input_type == "Text Description":
        input_data = st.text_area(
            "Describe the content or topic",
            placeholder="E.g. Chapter 3: Fractions for Class 5",
            key="test_creation_text"
        )
    elif input_type == "Voice Instruction (Text Proxy)":
        input_data = st.text_area(
            "Enter voice instruction as text",
            placeholder="E.g. Create a test on photosynthesis for Class 7",
            key="test_creation_voice"
        )
    else:
        input_data = st.text_input(
            "Enter URL",
            placeholder="E.g. https://example.com/science-lesson",
            key="test_creation_url"
        )
    
    context = st.text_area(
        "Additional Context",
        placeholder="E.g. Class 5, focus on fractions, include culturally relevant examples",
        key="test_creation_context"
    )
    num_questions = st.number_input(
        "Number of Questions",
        min_value=1,
        max_value=20,
        value=5,
        key="test_creation_num_questions"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        tts_language = st.selectbox(
            "Text-to-Speech Language",
            ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
            key="tts_language"
        )
    with col2:
        translate_to = st.selectbox(
            "Translate Questions To",
            ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
            key="translate_to"
        )
    
    if st.button("Generate Test", disabled=not input_data, key="test_creation_generate"):
        with st.spinner("Generating test..."):
            try:
                if input_type == "Textbook Photo":
                    content = process_image(input_data, use_ocr=True)
                    if "Error" in content:
                        st.error(content)
                        st.stop()
                else:
                    content = input_data
                
                st.write(f"**Debug Input Content**: {content}")
                
                prompt = f"""You are TestCraft, an AI agent for generating class-differentiated assessments. Create a test suite based on the provided input and context. Ensure questions are culturally relevant, tailored to student profiles (class level, difficulty), and include a mix of question types (MCQ, short answer, essay). Output **only** the test suite in this exact format, with no additional text, comments, or explanations:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4] (for MCQ, leave empty [] for others)
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]
- Question 2: ...

Example Output:
**Test Suite**:
- Question 1: If a farmer has 3/4 kg of rice and divides it among 3 children, how much does each get?
  Type: Short Answer
  Options: []
  Correct Answer: 1/4 kg
  Difficulty: Medium
- Question 2: Which crop is commonly grown in rural India?
  Type: MCQ
  Options: [Rice; Wheat; Corn; Potato]
  Correct Answer: Rice
  Difficulty: Easy
- Question 3: Explain the importance of the monsoon season in Indian agriculture.
  Type: Essay
  Options: []
  Correct Answer: [A detailed explanation about the monsoon's role]
  Difficulty: Hard

Input: {content}
Context: {context or 'No context provided'}, Class Level: {st.session_state.class_level}
Number of questions: {num_questions}

Instructions:
1. Incorporate hyper-local, culturally relevant examples (e.g., local crops like rice for biology, regional history for social studies).
2. Tailor questions to the specified class level and context.
3. Balance question types (at least one MCQ, one short answer, one essay if num_questions >= 3).
4. Ensure each question has a question text, type, options (if MCQ), correct answer, and difficulty.
5. Output **only** the test suite in the exact format shown in the example, with no additional text, headings, or explanations.
6. Use semicolons (;) to separate options for MCQs.
7. Ensure the output starts with "**Test Suite**:" and follows the exact structure shown.
8. For essay questions, provide a brief correct answer summarizing the expected response."""

                test_content = query_langchain(prompt)
                
                st.write("**Debug Raw LLM Output**:")
                st.code(test_content)
                
                questions = extract_questions_and_answers(test_content)
                
                if questions and "error" in questions[0]:
                    st.warning("Initial parsing failed. Retrying with simplified prompt...")
                    simplified_prompt = f"""You are TestCraft, an AI agent for generating assessments. Create a test suite with {num_questions} questions based on the input and context. Output **only** the test suite in this exact format, with no additional text:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4] (for MCQ, leave empty [] for others)
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]
- Question 2: ...

Example Output:
**Test Suite**:
- Question 1: If a farmer has 3/4 kg of rice and divides it among 3 children, how much does each get?
  Type: Short Answer
  Options: []
  Correct Answer: 1/4 kg
  Difficulty: Medium
- Question 2: Which crop is commonly grown in rural India?
  Type: MCQ
  Options: [Rice; Wheat; Corn; Potato]
  Correct Answer: Rice
  Difficulty: Easy

Input: {content}
Context: {context or 'No context provided'}, Class Level: {st.session_state.class_level}

Instructions:
1. Include culturally relevant examples.
2. Ensure each question has all fields: question text, type, options, correct answer, difficulty.
3. Output **only** the test suite in the exact format.
4. Use semicolons (;) for MCQ options."""
                    test_content = query_langchain(simplified_prompt, retry=True)
                    st.write("**Debug Retry LLM Output**:")
                    st.code(test_content)
                    questions = extract_questions_and_answers(test_content)
                
                if questions and "error" in questions[0]:
                    st.error(f"Failed to parse questions: {questions[0]['error']}")
                    st.stop()
                
                # Translate test suite if not English
                translated_questions = questions
                translated_test_content = test_content
                if translate_to != "English":
                    translated_questions = translate_test_suite(questions, translate_to)
                    # Reconstruct translated test content
                    translated_test_content = "**Test Suite**:\n"
                    for i, q in enumerate(translated_questions, 1):
                        translated_test_content += f"- Question {i}: {q['question']}\n"
                        translated_test_content += f"  Type: {q['type']}\n"
                        translated_test_content += f"  Options: [{'; '.join(q['options'])}]\n" if q['options'] else "  Options: []\n"
                        translated_test_content += f"  Correct Answer: {q['answer']}\n"
                        translated_test_content += f"  Difficulty: {q['difficulty']}\n"
                
                st.subheader("Generated Test")
                st.markdown(translated_test_content)
                
                # Multilingual Features
                if questions:
                    st.subheader("Multilingual Features")
                    selected_question = st.selectbox(
                        "Select a question to hear or translate",
                        [f"Q{i+1}: {q['question'][:50]}..." for i, q in enumerate(translated_questions)],
                        key="question_selector"
                    )
                    q_index = int(selected_question.split(":")[0][1:]) - 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔊 Listen to Question", key="tts_button"):
                            audio_base64 = text_to_speech(translated_questions[q_index]["question"], lang=tts_language)
                            if audio_base64:
                                play_audio(audio_base64)
                    with col2:
                        if st.button(f"Translate to {translate_to}", key="translate_button"):
                            lang_codes = {
                                "English": "en", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn",
                                "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
                                "Malayalam": "ml", "Punjabi": "pa"
                            }
                            translated_q = translate_text(translated_questions[q_index]["question"], target_lang=lang_codes.get(translate_to, "en"))
                            translated_opts = [translate_text(opt, target_lang=lang_codes.get(translate_to, "en")) for opt in translated_questions[q_index]["options"]]
                            translated_ans = translate_text(translated_questions[q_index]["answer"], target_lang=lang_codes.get(translate_to, "en"))
                            st.markdown(f"**Translated Question**: {translated_q}")
                            if translated_opts:
                                st.markdown(f"**Options**: {'; '.join(translated_opts)}")
                            st.markdown(f"**Correct Answer**: {translated_ans}")
                
                st.session_state.last_results = {
                    "type": "test",
                    "input_type": input_type,
                    "input_data": content,
                    "context": context,
                    "questions": translated_questions,
                    "test_content": translated_test_content,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.test_history.append(st.session_state.last_results)
                
            except Exception as e:
                st.error(f"Test generation failed: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")

# Hint Generator Tab (HintGuide Agent)
with tab2:
    st.subheader("Generate Hints")
    question = st.text_area(
        "Enter Question",
        placeholder="E.g. What is the capital of India?",
        key="hint_generator_question"
    )
    student_response = st.text_area(
        "Student Response (Optional)",
        placeholder="E.g. Mumbai",
        key="hint_generator_response"
    )
    
    if st.button("Generate Hint", disabled=not question, key="hint_generator_generate"):
        with st.spinner("Generating hint..."):
            try:
                prompt = f"""You are HintGuide, an AI agent providing progressive hints using Socratic questioning. Generate a hint for the given question and student response (if provided). Avoid giving direct answers, instead nudge the student toward the solution with a thought-provoking question or step-by-step guidance. Output **only** the hint in this format:

**Hint**: [Socratic question or step-by-step guidance]

Question: {question}
Student Response: {student_response or 'No response provided'}"""
                
                hint = query_langchain(prompt)
                # Translate hint to selected language
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
                st.error(f"Traceback: {traceback.format_exc()}")

# Grading Tab (GradeWise Agent)
with tab3:
    st.subheader("Grade Responses")
    response_type = st.radio(
        "Response Type",
        ["Text", "Handwritten Image"],
        key="grading_response_type"
    )
    responses = []
    
    if response_type == "Text":
        student_response = st.text_area(
            "Enter Student Response",
            placeholder="E.g. The capital of India is Delhi",
            key="grading_text_response"
        )
        correct_answer = st.text_input(
            "Correct Answer",
            placeholder="E.g. Delhi",
            key="grading_correct_answer"
        )
        rubric = st.text_area(
            "Grading Rubric",
            placeholder="E.g. 2 points for correct answer, 1 point for partial",
            key="grading_rubric"
        )
        
        if student_response:
            try:
                detected_lang = detect_language(student_response)
                st.write(f"Detected language: {detected_lang}")
                if detected_lang != "en":
                    translated_response = translate_text(student_response, target_lang="en")
                    st.write(f"Translated to English: {translated_response}")
                else:
                    translated_response = student_response
            except:
                st.warning("Language detection failed. Assuming English.")
                translated_response = student_response
        
        if st.button(
            "Grade Response",
            disabled=not (student_response and correct_answer and rubric),
            key="grading_generate_text"
        ):
            responses.append({
                "response": translated_response,
                "original_response": student_response,
                "correct_answer": correct_answer,
                "rubric": rubric
            })
    else:
        img_file = st.file_uploader(
            "Upload Handwritten Response",
            type=["jpg", "jpeg", "png"],
            key="grading_image"
        )
        correct_answer = st.text_input(
            "Correct Answer",
            key="grading_image_correct_answer"
        )
        rubric = st.text_area(
            "Grading Rubric",
            key="grading_image_rubric"
        )
        if img_file and st.button(
            "Grade Response",
            disabled=not (correct_answer and rubric),
            key="grading_generate_image"
        ):
            image = Image.open(img_file)
            st.image(image, width=300)
            description = process_image(image, use_ocr=True)
            if "Error" in description:
                st.error(description)
                st.stop()
            try:
                detected_lang = detect_language(description)
                st.write(f"Detected language: {detected_lang}")
                if detected_lang != "en":
                    translated_response = translate_text(description, target_lang="en")
                    st.write(f"Translated to English: {translated_response}")
                else:
                    translated_response = description
            except:
                st.warning("Language detection failed. Assuming English.")
                translated_response = description
            responses.append({
                "response": translated_response,
                "original_response": description,
                "correct_answer": correct_answer,
                "rubric": rubric
            })
    
    if responses:
        with st.spinner("Grading..."):
            try:
                prompt = f"""You are GradeWise, an AI agent for grading student responses. Grade the provided response based on the correct answer and rubric. Use sentiment analysis for essay responses and provide feedback in English. Return **only** the grade and feedback in this format:

**Grade**: [Score/Total]
**Feedback**: [Detailed feedback including what was correct, incorrect, and suggestions]

Response: {responses[0]['response']}
Correct Answer: {responses[0]['correct_answer']}
Rubric: {responses[0]['rubric']}"""
                
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
                st.error(f"Traceback: {traceback.format_exc()}")

# Analytics Tab (InsightTrack Agent)
with tab4:
    st.subheader("Learning Analytics")
    student_id = st.text_input(
        "Student ID",
        placeholder="E.g. STU001",
        key="analytics_student_id"
    )
    topic = st.text_input(
        "Topic",
        placeholder="E.g. Fractions",
        key="analytics_topic"
    )
    
    if st.button("Generate Insights", disabled=not (student_id and topic), key="analytics_generate"):
        with st.spinner("Generating insights..."):
            try:
                prompt = f"""You are InsightTrack, an AI agent for generating learning analytics. Analyze the student's performance and provide insights, including Learning Quotient (LQ) metrics, knowledge gaps, and remedial suggestions. Output **only** the analytics in this format:

**Learning Quotient (LQ)**:
- Accuracy: [X]%
- Time Taken: [X] minutes
- Hints Used: [X]
**Knowledge Gaps**: [Identified gaps]
**Remedial Suggestions**: [2-3 actionable suggestions]

Student ID: {student_id}
Topic: {topic}"""
                
                insights = query_langchain(prompt)
                translated_insights = insights
                if st.session_state.language != "English":
                    translated_insights = translate_text(insights, target_lang=st.session_state.language)
                st.markdown(translated_insights)
                
                student_data = {
                    "accuracy": 85,
                    "time_taken": 30,
                    "hints_used": 2
                }
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
                st.error(f"Analytics generation failed: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")
    
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
                        st.download_button(
                            "Download Report",
                            f,
                            file_name="sahayak_report.pdf",
                            mime="application/pdf",
                            key="analytics_download_report"
                        )
                    os.remove(pdf_path)
                    if st.session_state.test_history and st.session_state.test_history[-1].get("chart_path", "") and os.path.exists(st.session_state.test_history[-1]["chart_path"]):
                        os.remove(st.session_state.test_history[-1]["chart_path"])
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")

# Sidebar: Teacher Dashboard
with st.sidebar:
    st.header("Teacher Dashboard")
    st.subheader("Class Settings")
    class_level = st.selectbox(
        "Class Level",
        [f"Class {i}" for i in range(1, 13)],
        key="dashboard_class_level"
    )
    language = st.selectbox(
        "Feedback Language",
        ["English", "Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"],
        key="dashboard_language"
    )
    st.session_state.class_level = class_level
    st.session_state.language = language
    
    st.subheader("Test History")
    for entry in reversed(st.session_state.test_history):
        with st.expander(f"{entry['timestamp']} - {entry['type'].title()}", expanded=False):
            if entry['type'] == "test":
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

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ for Education • Powered by xAI</p>",
    unsafe_allow_html=True
)