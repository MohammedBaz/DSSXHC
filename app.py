import json
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Load Data and Initialize LLM ---
# Load the dataset (make sure hospital_data.json is in the same directory)
with open("hospital_data.json", "r") as f:
    hospital_data = json.load(f)

# Initialize the LLM (replace with your API key)
llm = OpenAI(openai_api_key="YOUR_API_KEY", temperature=0.2)

# --- Define Prompt Template ---
prompt_template = """
You are a hospital management consultant. Analyze the following data for {hospital_name}:

{hospital_data_str}

User's Question: {question}

Based on this data, provide a concise recommendation. Consider factors like occupancy rates, staffing levels, and length of stay.
Explain your reasoning.
"""

# --- Create LangChain Chain ---
prompt = PromptTemplate(
    input_variables=["hospital_name", "hospital_data_str", "question"],
    template=prompt_template
)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Streamlit Interface ---
st.title("Hospital Bed Capacity Advisor")

# Sidebar for Hospital Selection and Question
# Create a list of hospital names for the selectbox
hospital_names = [hospital["name"] for hospital in hospital_data["hospitals"]]
hospital_name = st.sidebar.selectbox("Select Hospital", hospital_names)

user_question = st.sidebar.text_area("Enter your question:", "Do you think it would be better to increase the bed capacity of hospital x to 100?")

# --- Get LLM Response ---
if st.sidebar.button("Get Recommendation"):
    # Find the selected hospital's data
    selected_hospital_data = None
    for hospital in hospital_data["hospitals"]:
        if hospital["name"] == hospital_name:
            selected_hospital_data = hospital
            break

    if selected_hospital_data is None:
        st.error(f"Data for {hospital_name} not found.")
    else:
        # Convert the selected hospital's data to a formatted string
        hospital_data_str = json.dumps(selected_hospital_data, indent=2)

        with st.spinner("Analyzing data and generating recommendation..."):
            response = chain.run(
                hospital_name=hospital_name,
                hospital_data_str=hospital_data_str,
                question=user_question
            )

        # --- Display Results ---
        st.subheader("Recommendation:")
        st.write(response)

        # --- Optional: Display Data ---
        if st.checkbox("Show Hospital Data"):
            st.json(hospital_data_str)
