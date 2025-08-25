import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def prompt_run(json_schema,system_message_str,user_message) -> any:
    # DEFINE YOUR PROMPT TEMPLATE AND SCHEMA
    json_schema_str =json_schema 
    json_schema = json.loads(json_schema_str)

    system_message = system_message_str
    
    user_message_template = user_message

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", user_message_template)
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    schema_for_llm = json_schema["schema"]
    schema_for_llm['title'] = json_schema.get('name', 'ResumeExtractor')
    schema_for_llm['description'] = "Extracts detailed competency and domain information from a resume."
    
    structured_llm = llm.with_structured_output(schema_for_llm)

    return prompt | structured_llm
    



load_dotenv()

def customization_prompt():
    json_schema = """{
        "name": "Course_Customisation_v1",
        "strict": false,
        "schema": {
            "type": "object",
            "properties": { "reasoning_overview": { "type": "string", "description": "High‑level overview of the task and description (max 100 words) of the approach or steps or logic you’ll take." }, "std_domain": { "type": "string", "description": "Mapped standard domain from the input." }, "std_competencies": { "type": "array", "description": "List of all competency names within scope of input, from the standard list of competencies that can be identified from the input, before building out level details.", "items": { "type": "string" }, "minItems": 6 }, "experience_level": { "type": "string", "description": "Target experience level of the competency based on years of experience and tasks done on job or project. If multiple or mixed, mention all that apply." }, "domain_id": { "type": "string", "description": "Auto‑generated ID: first three letters of each word in domain (plus optional names), plus an increment." }, "competencies": { "type": "array", "description": "Detailed entries for each competency (standard or unmapped).", "items": { "type": "object", "properties": { "competency_name": { "type": "string", "description": "Name of competency." }, "is_standard": { "type": "boolean", "description": "True if given competency exists in the standard list; false if the competency name is not present in the standard list." }, "importance_reasoning": { "type": "string", "description": "Reason out relevancy of the given competency with respect to overall resume or candidate profile." }, "importance": { "type": "string", "description": "Based on given details, mention the importance of the competency in terms of 'High', or 'Mid', or 'Low' with respect to input resume / Jd / Outline." }, "levels": { "type": "array", "description": "One entry per depth/level for this competency.", "items": { "type": "object", "properties": { "reasoning": { "type": "string", "description": "Facts listing and then detailed logic for this competency + level, including tools, methods, and processes and details mention either in the given in resume or assumed if not given. Also include on job experience in time terms." }, "level_name": { "type": "string", "description": "Identifier for this depth (e.g., 'Basic', 'Intermediate' or 'Advanced'), based on years of experience OR difficulty of tasks done on job or projects or course." }, "daily_tasks": { "type": "string", "description": "A single paragraph describing the daily tasks at this level for the competency." } }, "required": [ "reasoning", "level_name", "daily_tasks" ], "additionalProperties": false } } }, "required": [ "competency_name", "is_standard", "importance_reasoning", "importance", "levels" ], "additionalProperties": false }, "minItems": 1 }, "unmapped_items": { "type": "object", "description": "Any domains or competencies not in the standard lists, with reasoning.", "properties": { "reasoning": { "type": "string", "description": "Rationale for including these unmapped / mismatched domains or competencies if not in standard lists." }, "unmapped_domains": { "type": "string", "description": "List of domains that candidate is potentially relevant but not listed, if none then write 'n/a'." }, "unmapped_competencies": { "type": "array", "description": "List of competency names that were unmapped. if none 'n/a'.", "items": { "type": "string" }, "minItems": 0 } }, "required": [ "reasoning", "unmapped_domains", "unmapped_competencies" ], "additionalProperties": false } }, "required": [ "reasoning_overview", "std_domain", "std_competencies", "experience_level", "domain_id", "competencies", "unmapped_items" ], "additionalProperties": true } }
    """
    system_message ="You are an advanced AI system specialized in parsing unstructured candidate resumes and extracting various details as per 'guidelines for extraction' and schema."
    user_message ="""Extract various objects as per guidelines for extraction, various standard lists and other guidelines by applying them on Input given at the end.
Guidelines for extraction:
1. "Std_Domain": 1 most relevant domain inferred from title, summary, and responsibilities. Use the "standard list of domains"; if none fully fit, set unmapped_domain.
2. "Std_Competencies": All different and applicable competencies from the "standard list of competencies ", inferred from each work experience / project of the candidate and for relevant certificates or courses undertaken.
3. "ExperienceLevel": infer from any “years of experience” or level terms or history of candidate experience. using {{Basic, Intermediate, Advanced}}; otherwise “Unknown”.
4. "DailyTasks": For every competency, produce one cohesive paragraph (50–100 words) that describes the role’s day‑to‑day responsibilities, tailored to the specified proficiency level of the competency:
a. Anchor in Input
 – Base descriptions strictly on resume details. Include specific tech stack, tools, methods, etc.
 – Introduce external context only when it directly supports or clarifies a task, or very minimal details in work experience / job /projects / certificates are given.
b. Choose a Strong Lead Verb
 – Begin with a clear action (e.g., “Implement,” “Design,” "Evaluate", etc.).
 – Reflect the competency’s core purpose at the given level.
c. HR Focus (Non-Technical)
 - Ensure all topics are behavioral or situational, focusing on motivations, thought processes, learnings, and collaborative experiences. – Strictly avoid generating any technical topics that relate to  specific domain knowledge.  
d. Action & Process‑Orientation:
 – Frame each sentence around what the person does and why it matters.
 – Keep it present‑tense, active, and focused on outcomes.
e. Detailed & Distinctness:
 – Take an elaborate approach and maintain 15 words in description maximum.
 – Avoid repeating concepts across different competencies—each paragraph should cover unique responsibilities.
### Standard List of Domains:
{{Software Development, Data Science & Business Analytics, AI & Machine Learning, Cloud Computing & DevOps, Generative AI, Agile and Scrum, Digital Marketing, Cyber Security, IT Service and Architecture, Project Management, Product and Design, Hardware and Embedded Systems, Robotics and Automation, Mechanical and Structural Engineering, Manufacturing & Process Engineering, Finance & FinTech, Human Resources Management, Operations Management, Customer Support & Service, Quality Management, Business and Leadership, Supply Chain and Logistics, Sales and Business Development, Software Testing & Quality Assurance}}
### Standard List of Competencies:
{{Self-Presentation & Personal Branding ,Career Clarity & Goal Alignment ,Communication & Explanation Skills ,Project-work & Internships}}
### EXPERIENCE-LEVEL INFERENCE Guidelines:
- "Basic": 0–1 yrs or "junior" / “intern”/“trainee" / "entry-level" / 'req only up to 1 years of experience'.”  
- "Intermediate": 2–4 yrs or “intermediate-level" / 'req one full time experience'.”
- "Advanced": 5–6+ yrs or “senior”/“high-level" / "manager-level' / 'req worked in multiple roles'.”  
- Else: “Unknown”
Note: if sometimes the exp. range given is mixed / contains overlap in levels, list a description at each of the given levels.
### Unmapped Competencies guidelines:
- List major / important areas in work experience, projects, certifications or courses that are not labeled in std_competencies here.
Note: do not map very niche areas here, but important and little broad areas.
Note: up to 3 items max.
### Candidate Resume:
{resume_text}
"""
    return prompt_run(json_schema,system_message,user_message)
def blueprint_prompt():
  

    system_message = "You are an expert in aHR interviews . When given input data about a std_domain,  std_competencies, skills, experiencelevel, and dailytasks and other data, your job is to generate a structured list of subtopics (number as per user inputs), each with a tailored description. You must ground each subtopic in the intersection of:\n- each of std_competency,\n- the ExperienceLevel,\n- the Std_domain ,\n- the skills, and\n- the daily tasks.\nCarefully apply guidelines on the inputs given."
    user_message = """
    ## OBJECTIVE:
    For each listed competency, generate a list of 3–10 subtopics based on the guidelines for subtopic count, title rules and description rules and general rules.
    ## Guidelines:

    1. Subtopic Count:
    - Quantity: Generate number of sub topics per competency as follows.
    - Low Importance: Prepare 3 - 5 questions or probing points.
    - Mid Importance: Prepare 4 - 7 questions or probing points.
    - High Importance: Prepare 6 - 10 questions or probing points.
    - Daily-Task Mapping: Ensure your questions reflect the real-world challenges and responsibilities of the job for that specific career level (level_name), prompting stories that are relevant to our needs.If a competency is narrow (e.g., "Attention to Detail"), use fewer questions. If it's broad (e.g., "Strategic Thinking"), use more questions from the suggested range to explore its different facets.

    2. Title Rules:
    - This refers to the main behavioral prompt or question we ask the candidate.
    . Topic Naming & Clarity : Keep the topic name concise and clear, ideally under 7 words. The name should instantly communicate the competency being discussed.
    - Focus on Competency, Not Tasks : Define the topic around a core challenge, skill, or business outcome. Avoid specific technical tasks or jargon.
    Instead of: "Validating JWT Authentication"
    Use: "Ensuring System or Data Security"
    - Format as a Noun or Gerund Phrase : State the topic as a noun phrase, often using a gerund (a verb ending in "-ing"). This names the skill area without asking a direct question.
    Examples: "Managing Team Conflict," "Influencing Without Authority," "Driving a Project with Ambiguity."
    -  Design as a Conversation Starter : The topic should serve as a clear theme that allows an interviewer to easily frame a behavioral question (e.g., "Let's talk about [Topic Name]. Tell me about a time when..."). It primes the candidate to recall relevant stories.

    3. Description Rules:
    - These are the internal notes for the interviewer, defining what a strong answer looks like and what to listen for:
        a. Length: Keep the evaluation criteria to a single, focused sentence (around 15–20 words).
        b. Content: The description should specify the target behaviors we're assessing, such as problem-understanding, critical thinking, decision-making, or stakeholder influence.
        c. Depth Alignment: Tailor your expectations to the candidate’s experience level (ExperienceLevel). As seniority increases, look for a shift from foundational execution towards broader strategic ownership, handling ambiguity, and influencing outcomes. 
        d. Tone: Use impact-oriented verbs to describe the desired actions and results we want to hear about in the candidate's story (e.g., “diagnosed,” “influenced,” “streamlined,” “validated”). 
       e. Verbal-Only: Reiterate that the evaluation is based purely on the candidate's spoken response. 
       f. Concise: Your description should be dense with meaning, using commas to connect ideas rather than long sentences.  
       g. Domain Language: Translate the role’s daily tasks into broader, capability-based phrases that are relevant to the Std_Domain. 
       h. Subtopics: Ensure your set of questions for a competency provides varied coverage (e.g., for "Leadership," you might probe on delegation, conflict resolution, and mentoring). Avoid asking different questions that test the exact same behavior.

    6. Format your response in valid JSON.

    ## INPUT PARAMETERS:
    {input}
    """
    json_schema = """
    {
        "name": "Blueprint_generation_HR_v1",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "std_competencies_blueprint": {
                    "type": "array",
                    "description": "A list of competency subtopics",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "competency_name": { "type": "string", "description": "Name of the main competency" },
                            "competency_level": { "type": "string", "description": "level of the competency (Basic, Intermediate or Advanced)." },
                            "competency_id": { "type": "string", "description": "Take the first three letters of each word in the competency name, then append the next sequential number (i.e. current competency number + 1). example: <Wor…> <Wor…> … (N + 1)" },
                            "list_of_topics": {
                                "type": "array",
                                "description": "A list of subtopics related to the competency",
                                "minItems": 4,
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "topic_name": { "type": "string", "description": "Concise subtopic title (max N words as per user-defined constraint)" },
                                        "topic_description": { "type": "string", "description": "Brief, skill-aware description (max N words as per user-defined constraint)" },
                                        "topic_level": { "type": "string", "description": "depth / level at which the description is made." },
                                        "topic_id": { "type": "string", "description": "T-[number]" }
                                    },
                                    "required": [ "topic_name", "topic_description", "topic_level", "topic_id" ],
                                    "additionalProperties": false
                                }
                            }
                        },
                        "required": [ "competency_name", "competency_level", "competency_id", "list_of_topics" ],
                        "additionalProperties": false
                    }
                }
            },
            "required": [ "std_competencies_blueprint" ],
            "additionalProperties": false
        }
    }
    """
    return prompt_run(json_schema, system_message, user_message)
