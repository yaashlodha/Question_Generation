# input_streamlit.py

import streamlit as st
import json
import pandas as pd

# Import both of your prompt-generating functions
from customization_prompt import customization_prompt, blueprint_prompt

# --- PAGE 1: The HR Interview Workflow ---
def show_hr_interview_page():
    """
    This function contains the entire HR interview workflow we've already built.
    """
    st.header("HR Interview Blueprint Generator")

    # Add a "Back" button to return to the main selection
    if st.button("⬅️ Back to Selection"):
        st.session_state.page = "selection"
        st.rerun()

    # --- INITIALIZE CHAINS and SESSION STATE for this page ---
    try:
        analysis_chain = customization_prompt()
        blueprint_chain = blueprint_prompt()
    except Exception as e:
        st.error(f"Failed to initialize the analysis models: {e}")
        return

    # Use session_state to store the result between steps
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    # --- USER INPUT ---
    st.subheader("1. Paste Resume Text")
    raw_text = st.text_area(
        "Paste the full text of the resume here:",
        height=300,
        key="hr_resume_text" # Use a unique key for this text area
    )

    # --- STEP 1: INITIAL ANALYSIS ---
    if st.button("Analyze Resume", disabled=not raw_text):
        with st.spinner("Analyzing resume... Please wait."):
            try:
                result = analysis_chain.invoke({"resume_text": raw_text})
                st.session_state.analysis_result = result
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.session_state.analysis_result = None

    # --- STEP 2: DISPLAY RESULT and GENERATE BLUEPRINT ---
    if st.session_state.analysis_result:
        st.subheader("2. Structured Analysis Result")
        st.json(st.session_state.analysis_result)

        st.subheader("3. Generate Interview Blueprint")
        if st.button("Generate Blueprint"):
            with st.spinner("Generating interview blueprint..."):
                try:
                    input_for_blueprint = json.dumps(st.session_state.analysis_result, indent=2)
                    blueprint_output = blueprint_chain.invoke({"input": input_for_blueprint})

                    st.markdown("### Interview Blueprint Table")
                    table_data = []
                    for competency in blueprint_output.get('std_competencies_blueprint', []):
                        competency_name = competency.get('competency_name')
                        for topic in competency.get('list_of_topics', []):
                            table_data.append({
                                "Competency Name": competency_name,
                                "Topic No.": topic.get('topic_id'),
                                "Topic Name": topic.get('topic_name'),
                                "Topic Description": topic.get('topic_description'),
                                "Topic Level": topic.get('topic_level')
                            })
                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(df)
                    else:
                        st.warning("Could not generate table data from the output.")
                        st.json(blueprint_output)
                except Exception as e:
                    st.error(f"An error occurred during blueprint generation: {e}")

# --- PAGE 2: The "Work in Progress" page ---
def show_tech_interview_page():
    """
    This function displays the "Work in Progress" message.
    """
    st.header("Technical Interview Blueprint Generator")

    if st.button("⬅️ Back to Selection"):
        st.session_state.page = "selection"
        st.rerun()
    
    st.info("This feature is currently a work in progress. Please check back later! 🚧", icon="💡")
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2RzOW90c2hwdmNqMTBqYmFjaDExY3d1aWZ0d210dnV1cWUyZzY4eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l2Spjn4nQ5g7B2kYE/giphy.gif", caption="Under Construction")


# --- The Main App Router ---
def main_app():
    """
    This is the main function that runs the Streamlit app and acts as a router.
    """
    st.set_page_config(page_title="AI Interview Blueprint", page_icon="🤖")

    # Initialize session state for page navigation if it doesn't exist
    if "page" not in st.session_state:
        st.session_state.page = "selection"
        # Clear previous analysis results when starting fresh
        st.session_state.analysis_result = None

    # --- PAGE ROUTER LOGIC ---
    if st.session_state.page == "selection":
        st.title("AI-Powered Interview Blueprint Generator")
        st.markdown("Welcome! This tool uses AI to generate tailored interview questions from a resume.")
        st.markdown("---")
        
        st.subheader("Which type of interview would you like to prepare for?")
        
        # Create two columns for the buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗣️ HR Interview", use_container_width=True):
                st.session_state.page = "hr_interview"
                st.rerun() # Rerun the app to navigate to the new page

        with col2:
            if st.button("💻 Technical Interview", use_container_width=True):
                st.session_state.page = "tech_interview"
                st.rerun() # Rerun the app to navigate to the new page

    elif st.session_state.page == "hr_interview":
        show_hr_interview_page()

    elif st.session_state.page == "tech_interview":
        show_tech_interview_page()


# Run the main app
if __name__ == "__main__":
    main_app()