def get_test_creation_prompt(content: str, context: str, class_level: str, num_questions: int) -> str:
    """
    Generate the prompt for creating a test suite with culturally relevant questions.
    
    Args:
        content (str): Input content (from text, image, or URL).
        context (str): User-provided context for the test.
        class_level (str): Class level (e.g., 'Class 5').
        num_questions (int): Number of questions to generate.
    
    Returns:
        str: Formatted prompt for test suite generation.
    """
    context_used = context.strip() if context and context.strip() else 'No context provided'
    return f"""You are TestCraft, an AI agent for generating assessments. Create a test suite based on the input and context. Ensure questions are culturally relevant, tailored to the class level, and include a mix of types (MCQ, short answer, essay). For MCQs, provide exactly 4 options and a correct answer that matches one of the options. Output **only** the test suite in this format:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4]
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]
- Question 2: ...

Input: {content}
Context: {context_used}, Class Level: {class_level}
Number of questions: {num_questions}
Instructions:
1. Use culturally relevant examples (e.g., rice, mangoes, local festivals).
2. Tailor questions to {class_level} (e.g., simpler concepts for younger students; avoid advanced vocabulary for Class 1-3).
3. Include a mix of MCQ (at least 50%), short answer, and essay questions.
4. For MCQs, provide exactly 4 distinct options, and the correct answer must be one of them.
5. Ensure all fields (question, type, options, correct answer, difficulty) are filled.
6. Use semicolons to separate MCQ options.
7. Avoid vague or incomplete answers (e.g., 'Not provided')."""

def get_test_creation_retry_prompt(content: str, context: str, class_level: str, num_questions: int) -> str:
    """
    Generate a simplified prompt for retrying test suite generation when parsing fails.
    
    Args:
        content (str): Input content (from text, image, or URL).
        context (str): User-provided context for the test.
        class_level (str): Class level (e.g., 'Class 5').
        num_questions (int): Number of questions to generate.
    
    Returns:
        str: Formatted simplified prompt for test suite generation.
    """
    context_used = context.strip() if context and context.strip() else 'No context provided'
    return f"""You are TestCraft. Create a test suite with {num_questions} questions based on input and context. Output **only** the test suite in the exact format:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4]
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]

Input: {content}
Context: {context_used}, Class Level: {class_level}
Instructions:
1. Include culturally relevant examples (e.g., rice, mangoes, local festivals).
2. Tailor questions to {class_level} (e.g., simpler concepts for younger students).
3. Include a mix of MCQ (at least 50%), short answer, and essay questions.
4. For MCQs, provide exactly 4 distinct options, and the correct answer must be one of them.
5. Ensure all fields (question, type, options, correct answer, difficulty) are filled.
6. Use semicolons to separate MCQ options.
7. Avoid vague or incomplete answers (e.g., 'Not provided').
8. If parsing fails again, simplify further to basic questions."""

def get_hint_generator_prompt(question: str, student_response: str, hint_level: int = 2) -> str:
    """
    Generate the prompt for creating a Socratic hint based on a question and student response.
    
    Args:
        question (str): The question to generate a hint for.
        student_response (str): The student's response (optional).
        hint_level (int): Hint specificity level (1-3, default 2).
    
    Returns:
        str: Formatted prompt for hint generation.
    """
    return f"""You are HintGuide. Generate a hint for the question and response. Avoid direct answers, use Socratic questioning. Tailor hint complexity to level {hint_level} (1 = basic, 3 = detailed). Output **only**:

**Hint**: [Socratic question or guidance]

Question: {question}
Student Response: {student_response or 'No response provided'}"""

def get_grading_prompt(response: str, correct_answer: str, rubric: str) -> str:
    """
    Generate the prompt for grading a student response based on the correct answer and rubric.
    
    Args:
        response (str): The student's response.
        correct_answer (str): The correct answer.
        rubric (str): The grading rubric.
    
    Returns:
        str: Formatted prompt for grading.
    """
    return f"""You are GradeWise. Grade the response based on the correct answer and rubric. Provide feedback in English. Output **only**:

**Grade**: [Score/Total]
**Feedback**: [Feedback]

Response: {response}
Correct Answer: {correct_answer}
Rubric: {rubric or 'No rubric provided'}"""

def get_lesson_plan_prompt(class_level: str, subject: str, topic: str, duration: str, 
                         teaching_style: str, resources: list[str], language: str = "English") -> str:
    """
    Generate the prompt for creating a comprehensive lesson plan with teaching materials.
    
    Args:
        class_level (str): Class/Grade level (e.g., "Class 5")
        subject (str): Subject (e.g., "Mathematics")
        topic (str): Topic/Chapter (e.g., "Fractions")
        duration (str): Lesson duration (e.g., "1 hour")
        teaching_style (str): Preferred teaching style
        resources (List[str]): Available resources
        language (str): Language for materials
    
    Returns:
        str: Formatted prompt for lesson plan generation
    """
    return f"""You are LessonCraft, an AI agent for generating comprehensive lesson plans. Create a detailed lesson plan with teaching materials based on the provided parameters. Generate all text in {language}. Output **only** the lesson plan in this format:

**Lesson Plan for {class_level} {subject} - {topic}**
- Duration: {duration}
- Teaching Style: {teaching_style}
- Resources Available: {', '.join(resources) if resources else 'None specified'}
- Language: {language}

**Learning Objectives**:
1. [Objective 1]
2. [Objective 2]
3. [Objective 3]

**Lesson Structure**:
1. Warm-up Activity (5-10 mins): [Description]
2. Direct Instruction (15-20 mins): [Description]
3. Guided Practice (15-20 mins): [Description]
4. Independent Practice (10-15 mins): [Description]
5. Assessment/Closure (5-10 mins): [Description]

**Teaching Materials**:
- [Material 1 with description]
- [Material 2 with description]
- [Material 3 with description]

**Differentiation Strategies**:
- For struggling students: [Strategy]
- For advanced students: [Strategy]

**Homework/Extension Activities**:
1. [Activity 1]
2. [Activity 2]

**Notes**:
- Use culturally relevant examples (e.g., local festivals, foods)
- Include hands-on activities where possible
- Align with {class_level} curriculum standards
- Provide clear instructions for each activity
- Suggest low-cost alternatives for resource constraints"""

def get_analytics_prompt(student_id: str, topic: str, time_range: str = "All Time") -> str:
    """
    Generate the prompt for creating learning analytics based on student ID, topic, and time range.
    
    Args:
        student_id (str): The student's ID.
        topic (str): The topic for analysis.
        time_range (str): Time period for analysis (default "All Time").
    
    Returns:
        str: Formatted prompt for analytics generation.
    """
    return f"""You are InsightTrack. Provide analytics including Learning Quotient (LQ) metrics, gaps, and suggestions. Output **only**:

**Learning Quotient (LQ)**:
- Accuracy: [X]%
- Time Taken: [X] minutes
- Hints Used: [X]
**Knowledge Gaps**: [Gaps]
**Remedial Suggestions**: [Suggestions]

Student ID: {student_id}
Topic: {topic}
Time Range: {time_range}"""

def get_chatbot_prompt(query: str, user_role: str, class_level: str, chat_history: list[dict[str, str]]) -> str:
    """
    Generate the prompt for the Sahayak AI chatbot to answer educational queries.
    
    Args:
        query (str): The user's query.
        user_role (str): The role of the user (Teacher or Student).
        class_level (str): The class level (e.g., 'Class 5').
        chat_history (List[Dict[str, str]]): Previous messages in the conversation.
    
    Returns:
        str: Formatted prompt for chatbot response.
    """
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]]) if chat_history else "No previous context"
    return f"""You are SahayakAI, a friendly and knowledgeable AI assistant for the Sahayak Education Platform. Your role is to help {user_role.lower()}s with educational queries related to {class_level}. Provide clear, concise, and culturally relevant answers. Use examples like rice, mangoes, or local festivals where appropriate. Avoid direct answers for students unless necessary; encourage critical thinking. Output **only** the response:

Query: {query}
User Role: {user_role}
Class Level: {class_level}
Chat History: {history_context}
Instructions:
1. Tailor responses to {class_level} (e.g., simple language for younger students).
2. For teachers, offer pedagogical advice or resource suggestions.
3. For students, use Socratic questioning to guide learning when possible.
4. Keep responses under 200 words.
5. Translate to the user's preferred language if specified in context."""