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

# ------------------------ Setup ------------------------ #
st.set_page_config(page_title="📚 Sahayak: AI Education Platform", layout="wide")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it up.")
    st.stop()

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    """Initialize AI models without caching"""
    models = {}
    try:
        models['llm'] = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
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
if "grade_level" not in st.session_state:
    st.session_state.grade_level = "Grade 5"
if "language" not in st.session_state:
    st.session_state.language = "English"

# ------------------------ Helper Functions ------------------------ #
def query_langchain(prompt: str) -> str:
    """Query Groq LLM with error handling"""
    if not models['llm']:
        return "Error: LLM service unavailable"
    try:
        response = models['llm'].invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"

def process_image(image: Image.Image) -> str:
    """Extract text or describe content from images"""
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
    """Extract questions, answers, and metadata from LLM output with flexible parsing"""
    questions = []
    # Primary regex: strict format
    pattern = r'Question \d+: (.*?)\n(?:Type: (.*?)\n)?Options: (.*?)\nCorrect Answer: (.*?)\nDifficulty: (.*?)(?:\n|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question, q_type, options, answer, difficulty = match
        questions.append({
            "question": question.strip(),
            "type": q_type.strip() if q_type else "MCQ",
            "options": [opt.strip() for opt in options.split(";") if opt.strip()] if options.strip() else [],
            "answer": answer.strip(),
            "difficulty": difficulty.strip() if difficulty else "Medium"
        })
    
    # Fallback parsing if regex fails
    if not matches:
        # Try to extract any question-like content
        fallback_pattern = r'Question \d+: (.*?)(?:\n|$)(?:Type: (.*?)(?:\n|$))?(?:Options: (.*?)(?:\n|$))?(?:Correct Answer: (.*?)(?:\n|$))?(?:Difficulty: (.*?)(?:\n|$))?'
        fallback_matches = re.findall(fallback_pattern, text, re.DOTALL)
        for match in fallback_matches:
            question, q_type, options, answer, difficulty = match
            if question.strip():  # Only include if question text exists
                questions.append({
                    "question": question.strip(),
                    "type": q_type.strip() if q_type else "MCQ",
                    "options": [opt.strip() for opt in options.split(";") if opt.strip()] if options.strip() else [],
                    "answer": answer.strip() if answer else "Not provided",
                    "difficulty": difficulty.strip() if difficulty else "Medium"
                })
    
    if not questions:
        return [{"error": "No questions parsed from output"}]
    return questions

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
            placeholder="E.g. Chapter 3: Fractions for Grade 5",
            key="test_creation_text"
        )
    elif input_type == "Voice Instruction (Text Proxy)":
        input_data = st.text_area(
            "Enter voice instruction as text",
            placeholder="E.g. Create a test on photosynthesis for Grade 7",
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
        placeholder="E.g. Grade 5, focus on fractions, include culturally relevant examples",
        key="test_creation_context"
    )
    num_questions = st.number_input(
        "Number of Questions",
        min_value=1,
        max_value=20,
        value=5,
        key="test_creation_num_questions"
    )
    
    if st.button("Generate Test", disabled=not input_data, key="test_creation_generate"):
        with st.spinner("Generating test..."):
            try:
                if input_type == "Textbook Photo":
                    content = process_image(input_data)
                    if "Error" in content:
                        st.error(content)
                        st.stop()
                else:
                    content = input_data
                
                # Debug: Display input content
                st.write(f"**Debug Input Content**: {content}")
                
                prompt = f"""You are TestCraft, an AI agent for generating grade-differentiated assessments. Create a test suite based on the provided input and context. Ensure questions are culturally relevant, tailored to student profiles (grade level, difficulty), and include a mix of question types (MCQ, short answer, essay). Output **only** the test suite in this exact format, with no additional text or comments:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4] (for MCQ, leave empty for others)
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
Context: {context or 'No context provided'}, Grade Level: {st.session_state.grade_level}
Number of questions: {num_questions}

Instructions:
1. Incorporate hyper-local, culturally relevant examples (e.g., local crops like rice for biology, regional history for social studies).
2. Tailor questions to the specified grade level and context.
3. Balance question types (at least one MCQ, one short answer, one essay if num_questions >= 3).
4. Ensure each question has a question text, type, options (if MCQ), correct answer, and difficulty.
5. Output **only** the test suite in the exact format shown in the example, with no additional text, headings, or explanations."""
                
                test_content = query_langchain(prompt)
                
                # Debug: Display raw LLM output
                st.write("**Debug Raw LLM Output**:")
                st.code(test_content)
                
                questions = extract_questions_and_answers(test_content)
                if questions and "error" in questions[0]:
                    st.error(f"Failed to parse questions: {questions[0]['error']}")
                    st.stop()
                
                st.subheader("Generated Test")
                st.markdown(test_content)
                
                st.session_state.last_results = {
                    "type": "test",
                    "input_type": input_type,
                    "input_data": content,
                    "context": context,
                    "questions": questions,
                    "test_content": test_content,
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
        if st.button(
            "Grade Response",
            disabled=not (student_response and correct_answer and rubric),
            key="grading_generate_text"
        ):
            responses.append({"response": student_response, "correct_answer": correct_answer, "rubric": rubric})
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
            description = process_image(image)
            if "Error" in description:
                st.error(description)
                st.stop()
            responses.append({"response": description, "correct_answer": correct_answer, "rubric": rubric})
    
    if responses:
        with st.spinner("Grading..."):
            try:
                prompt = f"""You are GradeWise, an AI agent for grading student responses. Grade the provided response based on the correct answer and rubric. Use sentiment analysis for essay responses and provide feedback in {st.session_state.language}. Return **only** the grade and feedback in this format:

**Grade**: [Score/Total]
**Feedback**: [Detailed feedback including what was correct, incorrect, and suggestions]

Response: {responses[0]['response']}
Correct Answer: {responses[0]['correct_answer']}
Rubric: {responses[0]['rubric']}"""
                
                grading_result = query_langchain(prompt)
                st.markdown(grading_result)
                
                st.session_state.test_history.append({
                    "type": "grade",
                    "response": responses[0]["response"],
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
                st.markdown(insights)
                
                # Simulate student data for visualization
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
                    "insights": insights,
                    "chart_path": chart_path,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
            except Exception as e:
                st.error(f"Analytics generation failed: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")
    
    # Export report
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
    grade_level = st.selectbox(
        "Grade Level",
        ["Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7"],
        key="dashboard_grade_level"
    )
    language = st.selectbox(
        "Feedback Language",
        ["English", "Hindi", "Tamil", "Bengali"],
        key="dashboard_language"
    )
    st.session_state.grade_level = grade_level
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