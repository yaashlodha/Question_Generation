# AI Interview Blueprint and Question Generator 


This project is a web application built with Streamlit and LangChain that streamlines the initial stages of the hiring process. It takes a candidate's raw resume text, analyzes it to extract structured information, and then generates a tailored set of interview questions based on the analysis.



---

##  Features

* **Multi-Step AI Workflow**: Implements a two-stage process where the output of the first AI model (resume analysis) becomes the input for the second (interview question generation).
* **Structured Data Extraction**: Leverages LangChain's structured output capabilities to parse unstructured resume text into a predictable JSON format, extracting key details like domain, experience level, and competencies.
* **Dynamic Interview Blueprint**: Automatically generates relevant, competency-based interview questions tailored to the candidate's specific profile.
* **Interactive Web UI**: A user-friendly interface built with Streamlit that allows users to select an interview type, paste a resume, and view the results in a clean, tabular format.
* **Modular & Scalable**: The application is structured with separate "pages," making it easy to expand with new features (like the "Technical Interview" placeholder).

---

##  How It Works

The application follows a simple, three-step flow:

1.  **Select Interview Type**: The user chooses between an "HR Interview" or a "Technical Interview" (Work in progress).
2.  **Analyze Resume**: The user pastes the full text of a resume. The first LangChain prompt (`customization_prompt`) is invoked, which analyzes the text and outputs a structured JSON object.
3.  **Generate Blueprint**: The JSON output from the analysis is fed into the second LangChain prompt (`blueprint_prompt`). This prompt generates a detailed list of interview topics and descriptions, which are then displayed in an interactive table.

---

##  Tech Stack

* **Backend & AI**:
    * Python
    * [LangChain](https://www.langchain.com/) for orchestrating LLM prompts.
    * [OpenAI API](https://openai.com/blog/openai-api) (specifically `gpt-4o`) for language model processing.
* **Frontend**:
    * [Streamlit](https://streamlit.io/) for creating the interactive web interface.
* **Data Handling**:
    * [Pandas](https://pandas.pydata.org/) for data manipulation and creating the final table display.

---

##  Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yaashlodha/Question_Generation.git](https://github.com/yaashlodha/Question_Generation.git)
    cd Question_Generation
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create the environment
    python -m venv env

    # Activate it (Windows)
    .\env\Scripts\activate

    # Activate it (macOS/Linux)
    source env/bin/activate

    # Install required packages
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    * Create a new file named `.env` in the root of the project.
    * Add your OpenAI API key to this file:
        ```
        OPENAI_API_KEY="sk-YourSecretApiKeyHere"
        ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run input_streamlit.py
    ```
    The application should now be running and accessible in your web browser.

---

##  Future Improvements

* **Implement Technical Interview Workflow**: Build out the second prompt chain to generate technical questions based on the skills identified in the resume.
* **PDF/DOCX Upload**: Allow users to upload resume files directly instead of pasting text.
* **Error Handling**: Enhance error handling for API failures or malformed resume text.
* **Caching**: Implement Streamlit's caching to avoid re-running the AI models on the same resume text multiple times.
