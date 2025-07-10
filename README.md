# Sahayak AI Education Platform

![Sahayak Logo](https://img.icons8.com/color/48/000000/book.png)

**Version:** 1.0  
**Last Updated:** July 10, 2025, 01:24 PM IST  
**Developed by:** AI for Education Initiative  

Welcome to **Sahayak**, an AI-powered education platform designed to empower teachers and students in low-resource classrooms. Built with Streamlit and integrated with advanced language models (via Groq API), Sahayak offers tools for test creation, hint generation, grading, analytics, lesson planning, and a conversational chatbot. This platform supports multiple languages and culturally relevant content to enhance learning experiences.

---

## 📚 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

---

## 🚀 Features
- **1. 📝 Test Creation**: Generate culturally relevant assessments (MCQs, short answers, essays) from text, images, or URLs, with translation and audio support.
- **2. 📚 Class Prep**: Create detailed lesson plans with materials, tailored to class level and resources.
- **3. 🤖 Chat with Sahayak**: Interact with an AI chatbot for educational queries, supporting teachers and students.
- **4. 💡 Hint Generator**: Provide Socratic hints tailored to student responses and class levels.
- **5. ✅ Grading**: Automate grading of text or handwritten responses with feedback, supporting multiple languages.
- **6. 📊 Analytics**: Analyze student performance with Learning Quotient (LQ) metrics and visual charts.
- **🌐 Multi-Language Support**: Translate content into English, Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, and Punjabi.
- **🎨 Theme Switcher**: Switch between light and dark themes for better usability.
- **🔊 Audio Playback**: Listen to generated content via text-to-speech.

---

## 🛠️ Installation

### Prerequisites
- **Python 3.9+**
- **pip**
- **Git** (optional)

### Steps

#### 1. Clone the Repository (optional)
```bash
git clone https://github.com/your-username/sahayak-ai.git
cd sahayak-ai
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, create it with:
```
streamlit==1.30.0
langchain-groq==0.1.0
transformers==4.41.0
torch==2.3.0
matplotlib==3.8.0
fpdf==1.7.2
deep-translator==1.11.4
gtts==2.5.1
Pillow==10.2.0
pytesseract==0.3.10
playwright==1.42.0
python-dotenv==1.0.1
```

#### 3. Set Up Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

#### 4. Install Tesseract OCR
- Ubuntu: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: [Download here](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH

#### 5. Install Playwright Browsers
```bash
playwright install
```

#### 6. Run the Application
```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💻 Usage

### Interface Overview
- **Tabs**: Navigate using top-level numbered tabs.
- **Sidebar**: Adjust class settings, themes, view activity history.
- **Status Bar**: Displays real-time timestamp (e.g., 2025-07-10 13:24 IST).

### Step-by-Step Guide

#### 📝 Test Creation
- Upload a textbook image, enter text, or paste a URL.
- Add context (optional), select question count.
- Choose a translation language and generate questions.

#### 💡 Hint Generator
- Input question and student response (optional).
- Choose hint specificity level (1–3).
- Generate hint using AI.

#### ✅ Grading
- Submit typed or handwritten student response.
- Provide answer key and rubric (optional).
- AI will score and give feedback.

#### 📊 Analytics
- Enter student ID, topic, and time range.
- View performance graphs and LQ metrics.

#### 📚 Class Prep
- Enter class level, subject, topic, time, style, and resources.
- Get structured lesson plan with materials.

#### 🤖 Chat with Sahayak
- Select role (Teacher/Student), set class level.
- Ask questions and receive interactive AI responses.

### 🔧 Additional Features
- **PDF Export**: Download reports and test results.
- **Audio Support**: Text-to-speech for chatbot or questions.
- **History**: Browse previous sessions in sidebar.

---

## 🗂️ File Structure
```
project_directory/
├── README.md
├── app.py              # Main Streamlit app
├── light_theme.css     # Light mode styling
├── dark_theme.css      # Dark mode styling
├── prompts.py          # Prompt templates for AI
├── .env                # Groq API key
├── requirements.txt    # Python dependencies
```

---

## ⚙️ Configuration

- **API Key**: Set `GROQ_API_KEY` in `.env`
- **Language**: Set default language in the sidebar
- **Theme**: Use sidebar toggle to switch light/dark mode

---

## 🧠 Technical Details

- **Framework**: [Streamlit](https://streamlit.io/)
- **LLM Integration**: Groq `gemma2-9b-it` via `langchain-groq`
- **Image Captioning**: `Salesforce/blip-image-captioning-base`
- **OCR**: [Tesseract OCR](https://github.com/tesseract-ocr)
- **Translation**: `deep-translator` (Google Translate backend)
- **Text-to-Speech**: [gTTS](https://pypi.org/project/gTTS/)
- **PDF Generation**: `fpdf`
- **Web Scraping**: [Playwright](https://playwright.dev/)
- **Styling**: Custom CSS (`light_theme.css`, `dark_theme.css`)
- **Session Management**: `st.session_state` in Streamlit

---

## 🤝 Contributing

We welcome contributions!

### Steps:
1. **Fork** this repo
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Added feature XYZ"
   ```
4. Push to GitHub:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request describing your changes.

### Guidelines
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Write tests for new features
- Update this README if needed

---

## 🛠️ Troubleshooting

- **App Not Starting**: Check API key and all dependencies.
- **OCR Issues**: Ensure Tesseract is correctly installed and in PATH.
- **Web Extraction Fails**: Check network and URL correctness.
- **Chatbot Errors**: Recheck model loading in `app.py`.
- **Styling Problems**: Clear browser cache and reload.

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for full details.

---

## 📬 Contact

- **Email**: support@sahayak.ai  
- **GitHub Issues**: [Submit an Issue](https://github.com/your-username/sahayak-ai/issues)

---

**Happy teaching and learning with Sahayak! 🎓**
