def get_test_creation_prompt(content: str, context: str, class_level: str, num_questions: int) -> str:
    return f"""You are TestCraft, an AI agent for generating assessments. Create a test suite based on the input and context. Ensure questions are culturally relevant and include a mix of types (MCQ, short answer, essay). Output **only** the test suite in this format:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4]
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]
- Question 2: ...

Input: {content}
Context: {context or 'No context provided'}, Class Level: {class_level}
Number of questions: {num_questions}
Instructions:
1. Use culturally relevant examples (e.g., rice, mangoes).
2. Tailor to class level and context.
3. Balance question types.
4. Ensure all fields are included.
5. Use semicolons for MCQ options."""

def get_test_creation_retry_prompt(content: str, context: str, class_level: str, num_questions: int) -> str:
    return f"""You are TestCraft. Create a test suite with {num_questions} questions based on input and context. Output **only** the test suite in the exact format:

**Test Suite**:
- Question 1: [Question text]
  Type: [MCQ/Short Answer/Essay]
  Options: [Option1; Option2; Option3; Option4]
  Correct Answer: [Answer]
  Difficulty: [Easy/Medium/Hard]

Input: {content}
Context: {context or 'No context provided'}, Class Level: {class_level}
Instructions:
1. Include culturally relevant examples.
2. Ensure all fields are included.
3. Use semicolons for MCQ options."""

def get_hint_generator_prompt(question: str, student_response: str) -> str:
    return f"""You are HintGuide. Generate a hint for the question and response. Avoid direct answers, use Socratic questioning. Output **only**:

**Hint**: [Socratic question or guidance]

Question: {question}
Student Response: {student_response or 'No response provided'}"""

def get_grading_prompt(response: str, correct_answer: str, rubric: str) -> str:
    return f"""You are GradeWise. Grade the response based on the correct answer and rubric. Provide feedback in English. Output **only**:

**Grade**: [Score/Total]
**Feedback**: [Feedback]

Response: {response}
Correct Answer: {correct_answer}
Rubric: {rubric}"""

def get_analytics_prompt(student_id: str, topic: str) -> str:
    return f"""You are InsightTrack. Provide analytics including Learning Quotient (LQ) metrics, gaps, and suggestions. Output **only**:

**Learning Quotient (LQ)**:
- Accuracy: [X]%
- Time Taken: [X] minutes
- Hints Used: [X]
**Knowledge Gaps**: [Gaps]
**Remedial Suggestions**: [Suggestions]

Student ID: {student_id}
Topic: {topic}"""