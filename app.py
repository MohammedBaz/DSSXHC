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
llm = OpenAI(openai_api_key=st.secrets["OpenAIKey"], temperature=0.2)

# --- Define Prompt Template ---
prompt_template = """
You are a hospital management consultant. Analyze the following data for Hospital X:

Bed Capacity: {bed_capacity}
Occupancy Rate: {occupancy_rate}
Surgery Department:
  - Occupancy: {surgery_occupancy}%
  - Average Stay: {surgery_avg_stay} days
  - Doctors: {surgery_doctors}
  - Nurses: {surgery_nurses}

User's Question: {question}

Based on this data, provide a concise recommendation. Consider factors like occupancy rates, staffing levels, and length of stay.
Explain your reasoning, focusing especially on the surgery department's high occupancy and short stay.
"""

# --- Create LangChain Chain ---
prompt = PromptTemplate(
    input_variables=["bed_capacity", "occupancy_rate", "surgery_occupancy", "surgery_avg_stay", "surgery_doctors", "surgery_nurses", "question"],
    template=prompt_template
)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Streamlit Interface ---
st.title("Hospital Bed Capacity Advisor")

# Sidebar for Hospital Selection and Question
hospital_name = st.sidebar.selectbox("Select Hospital", list(hospital_data.keys()))
user_question = st.sidebar.text_area("Enter your question:", "Do you think it would be better to increase the bed capacity of hospital x to 100?")

# --- Get LLM Response ---
if st.sidebar.button("Get Recommendation"):
    data = hospital_data[hospital_name]
    surgery_data = data["departments"]["surgery"]

    with st.spinner("Analyzing data and generating recommendation..."):
        response = chain.run(
            bed_capacity=data["bed_capacity"],
            occupancy_rate=data["occupancy_rate"],
            surgery_occupancy=surgery_data["occupancy_percentage"],
            surgery_avg_stay=surgery_data["avg_stay_days"],
            surgery_doctors=surgery_data["doctors"],
            surgery_nurses=surgery_data["nurses"],
            question=user_question
        )

    # --- Display Results ---
    st.subheader("Recommendation:")
    st.write(response)

    # --- Optional: Display Data ---
    if st.checkbox("Show Hospital Data"):
        st.write(data)
