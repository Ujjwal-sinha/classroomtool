import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
from datetime import datetime, date
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import re
import torch
import pandas as pd
import numpy as np
import base64
import json
import traceback

# ------------------------ Setup ------------------------ #
st.set_page_config(page_title="📚 Sahayak: AI Education Platform", layout="wide")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it up.")
    st.stop()

# Offline cache directory
CACHE_DIR = "sahayak_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ------------------------ Model Initialization ------------------------ #
@st.cache_resource
def load_models():
    """Initialize AI models with offline support"""
    models = {}
    
    # Initialize LangChain ChatGroq LLM
    try:
        models['llm'] = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        models['llm'] = None

    # Initialize BLIP for image processing (textbook photos)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        models['processor'] = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=CACHE_DIR
        )
        models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        ).to(device)
        models['blip_model'].eval()
    except Exception as e:
        st.error(f"Failed to load image processor: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
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

# ------------------------ Helper Functions ------------------------ #
def query_langchain(prompt):
    """Query Groq LLM with error handling and offline fallback"""
    if not models['llm']:
        return "LLM service unavailable"
    try:
        response = models['llm']([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"

def process_image(image: Image.Image) -> str:
    """Extract text or describe content from textbook images"""
    if not models['processor'] or not models['blip_model']:
        return "Image processing unavailable"
    
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        device = next(models['blip_model'].parameters()).device
        inputs = models['processor'](image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = models['blip_model'].generate(**inputs, max_new_tokens=100)
        description = models['processor'].decode(outputs[0], skip_special_tokens=True)
        return description
    except Exception as e:
        return f"Image processing error: {str(e)}"

def extract_questions_and_answers(text):
    """Extract questions, answers, and metadata from LLM output"""
    questions = []
    pattern = r'Question \d+: (.*?)\n(?:Type: (.*?)\n)?Options: (.*?)\nCorrect Answer: (.*?)\nDifficulty: (.*?)\n'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question, q_type, options, answer, difficulty = match
        questions.append({
            "question": question.strip(),
            "type": q_type.strip() if q_type else "MCQ",
            "options": [opt.strip() for opt in options.split(";") if opt.strip()],
            "answer": answer.strip(),
            "difficulty": difficulty.strip()
        })
    
    return questions

def generate_lq_chart(student_data):
    """Generate Learning Quotient chart for student performance"""
    if not student_data:
        return None
    
    metrics = ["Accuracy", "Time Taken (min)", "Hints Used"]
    values = [
        student_data.get("accuracy", 0),
        student_data.get("time_taken", 0),
        student_data.get("hints_used", 0)
    ]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics, values, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_title("Learning Quotient (LQ) Metrics")
    ax.set_ylabel("Value")
    plt.tight_layout()
    return fig

def generate_performance_heatmap(student_data):
    """Generate heatmap for class performance across topics"""
    topics = list(student_data.keys())
    students = list(student_data[topics[0]].keys()) if topics else []
    if not topics or not students:
        return None
    
    data = [[student_data[topic][student]["accuracy"] for student in students] for topic in topics]
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(data, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(students)))
    ax.set_yticks(np.arange(len(topics)))
    ax.set_xticklabels(students, rotation=45)
    ax.set_yticklabels(topics)
    plt.colorbar(heatmap, label="Accuracy (%)")
    ax.set_title("Class Performance Heatmap")
    plt.tight_layout()
    return fig

def generate_pdf_report(questions, analysis, chart, heatmap=None):
    """Generate PDF report with test details, analysis, and charts"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Sahayak Assessment Report", ln=1, align="C")
        pdf.set_font("Arial", "", 12)
        
        # Add questions
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Generated Questions", ln=1)
        pdf.set_font("Arial", "", 10)
        for q in questions:
            pdf.multi_cell(0, 8, f"Question: {q['question']}")
            pdf.cell(0, 8, f"Type: {q['type']}", ln=1)
            pdf.cell(0, 8, f"Options: {'; '.join(q['options'])}", ln=1)
            pdf.cell(0, 8, f"Correct Answer: {q['answer']}", ln=1)
            pdf.cell(0, 8, f"Difficulty: {q['difficulty']}", ln=1)
            pdf.ln(5)
        
        # Add analysis
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analysis", ln=1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 8, analysis)
        pdf.ln(10)
        
        # Add charts
        if chart:
            chart_path = "temp_chart.png"
            chart.savefig(chart_path, bbox_inches="tight", dpi=100)
            pdf.image(chart_path, w=180)
            os.remove(chart_path)
        
        if heatmap:
            heatmap_path = "temp_heatmap.png"
            heatmap.savefig(heatmap_path, bbox_inches="tight", dpi=100)
            pdf.image(heatmap_path, w=180)
            os.remove(heatmap_path)
        
        # Add footer with current date and time
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} IST", 0, 0, "C")
        
        pdf_path = "sahayak_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

# ------------------------ Streamlit UI ------------------------ #
st.title("📚 Sahayak: AI-Powered Education Platform")
st.caption("Empowering teachers in multi-grade, low-resource classrooms with automated assessments and analytics")
st.markdown(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST")

# Initialize tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Test Creation", "💡 Hint Generator", "✅ Grading", "📊 Analytics"])

# Test Creation Tab (TestCraft Agent)
with tab1:
    st.subheader("Create Assessments")
    input_type = st.radio("Input Type", ["Textbook Photo", "Text Description", "Voice Instruction (Text Proxy)", "URL"])
    input_data = None
    
    if input_type == "Textbook Photo":
        img_file = st.file_uploader("Upload textbook page", type=["jpg", "jpeg", "png"])
        if img_file:
            input_data = Image.open(img_file)
            st.image(input_data, use_column_width=True)
    elif input_type == "Text Description":
        input_data = st.text_area("Describe the content or topic", placeholder="E.g. Chapter 3: Fractions for Grade 5")
    elif input_type == "Voice Instruction (Text Proxy)":
        input_data = st.text_area("Enter voice instruction as text", placeholder="E.g. Create a test on photosynthesis for Grade 7")
    else:
        input_data = st.text_input("Enter URL", placeholder="E.g. https://example.com/science-lesson")
    
    context = st.text_area("Additional Context", placeholder="E.g. Grade 5, focus on fractions, include culturally relevant examples")
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
    
    if st.button("Generate Test", disabled=not input_data):
        with st.spinner("Generating test..."):
            try:
                # Process input based on type
                if input_type == "Textbook Photo":
                    content = process_image(input_data)
                    if "error" in content.lower():
                        st.error(content)
                        st.stop()
                else:
                    content = input_data
                
                prompt = f"""You are TestCraft, an AI agent for generating grade-differentiated assessments. Create a test suite based on the provided input and context. Ensure questions are culturally relevant, tailored to student profiles (grade level, difficulty), and include a mix of question types (MCQ, short answer, essay). Follow this format:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4] (for MCQ)
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]
- Question 2: ...

Input: {content}
Context: {context or 'No context provided'}
Number of questions: {num_questions}

Instructions:
1. Incorporate hyper-local, culturally relevant examples (e.g., local crops for biology, regional history for social studies).
2. Tailor questions to the specified grade level and context.
3. Ensure offline compatibility by providing clear, self-contained questions.
4. Balance question types and difficulty levels."""
                
                test_content = query_langchain(prompt)
                questions = extract_questions_and_answers(test_content)
                
                st.subheader("Generated Test")
                st.markdown(test_content)
                
                # Save results
                st.session_state.last_results = {
                    "type": "test",
                    "input_type": input_type,
                    "input_data": input_data if input_type != "Textbook Photo" else content,
                    "context": context,
                    "questions": questions,
                    "test_content": test_content,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.test_history.append(st.session_state.last_results)
                
            except Exception as e:
                st.error(f"Test generation failed: {str(e)}")

# Hint Generator Tab (HintGuide Agent)
with tab2:
    st.subheader("Generate Hints")
    question = st.text_area("Enter Question", placeholder="E.g. What is the capital of India?")
    student_response = st.text_area("Student Response (Optional)", placeholder="E.g. Mumbai")
    
    if st.button("Generate Hint"):
        with st.spinner("Generating hint..."):
            prompt = f"""You are HintGuide, an AI agent providing progressive hints using Socratic questioning. Generate a hint for the given question and student response (if provided). Avoid giving direct answers, instead nudge the student toward the solution with a thought-provoking question or step-by-step guidance.

Question: {question}
Student Response: {student_response or 'No response provided'}

Provide the hint in this format:
**Hint**: [Socratic question or step-by-step guidance]"""
            
            hint = query_langchain(prompt)
            st.markdown(hint)
            
            # Save to history
            st.session_state.test_history.append({
                "type": "hint",
                "question": question,
                "student_response": student_response,
                "hint": hint,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

# Grading Tab (GradeWise Agent)
with tab3:
    st.subheader("Grade Responses")
    response_type = st.radio("Response Type", ["Text", "Handwritten Image"])
    responses = []
    
    if response_type == "Text":
        student_response = st.text_area("Enter Student Response", placeholder="E.g. The capital of India is Delhi")
        correct_answer = st.text_input("Correct Answer", placeholder="E.g. Delhi")
        rubric = st.text_area("Grading Rubric", placeholder="E.g. 2 points for correct answer, 1 point for partial")
        if st.button("Grade Response"):
            responses.append({"response": student_response, "correct_answer": correct_answer, "rubric": rubric})
    else:
        img_file = st.file_uploader("Upload Handwritten Response", type=["jpg", "jpeg", "png"])
        correct_answer = st.text_input("Correct Answer")
        rubric = st.text_area("Grading Rubric")
        if img_file and st.button("Grade Response"):
            image = Image.open(img_file)
            st.image(image, width=300)
            description = process_image(image)
            responses.append({"response": description, "correct_answer": correct_answer, "rubric": rubric})
    
    if responses:
        with st.spinner("Grading..."):
            prompt = f"""You are GradeWise, an AI agent for grading student responses. Grade the provided response based on the correct answer and rubric. Use sentiment analysis for essay responses and allow for teacher review. Return the grade and feedback in this format:

**Grade**: [Score/Total]
**Feedback**: [Detailed feedback including what was correct, incorrect, and suggestions]

Response: {responses[0]['response']}
Correct Answer: {responses[0]['correct_answer']}
Rubric: {responses[0]['rubric']}"""
            
            grading_result = query_langchain(prompt)
            st.markdown(grading_result)
            
            # Save to history
            st.session_state.test_history.append({
                "type": "grade",
                "response": responses[0]["response"],
                "correct_answer": responses[0]["correct_answer"],
                "rubric": responses[0]["rubric"],
                "grading_result": grading_result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

# Analytics Tab (InsightTrack Agent)
with tab4:
    st.subheader("Learning Analytics")
    student_id = st.text_input("Student ID", placeholder="E.g. STU001")
    topic = st.text_input("Topic", placeholder="E.g. Fractions")
    
    if st.button("Generate Insights"):
        with st.spinner("Generating insights..."):
            prompt = f"""You are InsightTrack, an AI agent for generating learning analytics. Analyze the student's performance based on their test history and provide insights, including Learning Quotient (LQ) metrics, knowledge gaps, and remedial suggestions. Use the following format:

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
                "accuracy": 85,  # Example values
                "time_taken": 30,
                "hints_used": 2
            }
            chart = generate_lq_chart(student_data)
            if chart:
                st.pyplot(chart)
            
            # Simulate class data for heatmap
            class_data = {
                topic: {
                    "STU001": {"accuracy": 85},
                    "STU002": {"accuracy": 70},
                    "STU003": {"accuracy": 90}
                }
            }
            heatmap = generate_performance_heatmap(class_data)
            if heatmap:
                st.pyplot(heatmap)
            
            # Save to history
            st.session_state.test_history.append({
                "type": "insights",
                "student_id": student_id,
                "topic": topic,
                "insights": insights,
                "chart": chart,
                "heatmap": heatmap,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
    
    # Export report
    if st.session_state.last_results.get("type") == "test":
        if st.button("📄 Generate Test Report"):
            pdf_path = generate_pdf_report(
                st.session_state.last_results.get("questions", []),
                st.session_state.last_results.get("test_content", ""),
                st.session_state.test_history[-1].get("chart") if st.session_state.test_history else None,
                st.session_state.test_history[-1].get("heatmap") if st.session_state.test_history else None
            )
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download Report",
                        f,
                        file_name="sahayak_report.pdf",
                        mime="application/pdf"
                    )
                os.remove(pdf_path)

# Sidebar: Teacher Dashboard
with st.sidebar:
    st.header("Teacher Dashboard")
    
    st.subheader("Class Settings")
    grade_level = st.selectbox("Grade Level", ["Grade 3", "Grade 4", "Grade 5", "Grade 6", "Grade 7"])
    language = st.selectbox("Feedback Language", ["English", "Hindi", "Tamil", "Bengali"])
    st.session_state.grade_level = grade_level
    st.session_state.language = language
    
    st.subheader("Test History")
    for entry in reversed(st.session_state.test_history):
        with st.expander(f"{entry['timestamp']} - {entry['type'].title()}"):
            if entry['type'] == "test":
                st.markdown(entry["test_content"])
            elif entry['type'] == "hint":
                st.markdown(f"**Question**: {entry['question']}\n**Hint**: {entry['hint']}")
            elif entry['type'] == "grade":
                st.markdown(entry["grading_result"])
            elif entry['type'] == "insights":
                st.markdown(entry["insights"])
                if entry.get("chart"):
                    st.pyplot(entry["chart"])
                if entry.get("heatmap"):
                    st.pyplot(entry["heatmap"])
    
    if st.button("Clear History"):
        st.session_state.test_history.clear()
        st.session_state.last_results = {}
        st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ for Education • Powered by xAI</p>",
    unsafe_allow_html=True
)