import json
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
import re

# --- Load Data and Initialize LLM ---
with open("hospital_data.json", "r") as f:
    hospital_data = json.load(f)

llm = OpenAI(openai_api_key=st.secrets["OpenAIKey"], temperature=0.2)  # General questions
chat_llm = ChatOpenAI(openai_api_key=st.secrets["OpenAIKey"], temperature=0.2)  # Data analysis

# --- Define Prompt Templates ---
# General Question Prompt Template
general_prompt_template = """
You are a helpful AI assistant. Answer the following question as concisely as possible:

Question: {question}
"""
general_prompt = PromptTemplate(
    input_variables=["question"],
    template=general_prompt_template
)
general_chain = LLMChain(llm=llm, prompt=general_prompt)

# Classification Prompt Template
classification_prompt_template = """
You are an AI that classifies questions based on whether they require a detailed analysis of hospital data or not.
Classify the following question:

Question: {question}

Answer with 'simple' if the question can be answered directly without data analysis, or 'analysis' if it requires detailed data analysis.

Answer:
"""
classification_prompt = PromptTemplate(
    input_variables=["question"],
    template=classification_prompt_template
)
classification_chain = LLMChain(llm=llm, prompt=classification_prompt)

# Hospital Analysis Prompt Template
system_message_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are a hospital management consultant. Analyze the following data from these hospitals:
    {hospital_data_str}
    """
)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    """
    {question}

    Format your response with the following structure:

    ## Analysis:

    *   Present a detailed analysis of the provided data, including key metrics and calculations.
    *   Use bullet points to highlight important observations.

    ## Considerations:

    *   Discuss factors that should be taken into account when making recommendations, including potential limitations of the data.
    *   Use bullet points to list these considerations.

    ## Conclusion:

    *   Provide a clear and concise conclusion based on your analysis and considerations.

    ## Recommendations:

    *   Offer specific, actionable recommendations for the hospital.
    *   Use numbered points for each recommendation.

    ## Further Considerations (Optional):

    *   If applicable, suggest additional data or analysis that could provide further insights.
    """
)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
analysis_chain = LLMChain(llm=chat_llm, prompt=chat_prompt)

# --- Streamlit Interface ---
st.title("Healthcare Advisor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your question here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Classify the question type
    question_type = classification_chain.run(question=prompt).lower()

    # Handle simple questions
    if question_type == "simple":
        match = re.search(r"hospital\s*(\d+)", prompt, re.IGNORECASE)
        if match:
            hospital_name = f"Hospital{match.group(1)}"
            response = None
            for hospital in hospital_data["hospitals"]:
                if hospital["name"].lower() == hospital_name.lower():
                    response = f"{hospital_name} is located at {hospital['location']}."
                    break
            if not response:
                response = f"Could not find information for {hospital_name}."
        else:
            response = "I couldn't identify the hospital you're asking about."
        
        # Display response
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Handle complex questions
    elif question_type == "analysis":
        match = re.search(r"hospital\s*(\d+)", prompt, re.IGNORECASE)
        if match:
            hospital_name = f"Hospital{match.group(1)}"
            hospital_data_str = json.dumps(hospital_data, indent=2)

            # Find the selected hospital's data
            selected_hospital_data = None
            for hospital in hospital_data["hospitals"]:
                if hospital["name"] == hospital_name:
                    selected_hospital_data = hospital
                    break

            if selected_hospital_data:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data and generating recommendations..."):
                        response = analysis_chain.run(
                            hospital_name=hospital_name,
                            hospital_data_str=hospital_data_str,
                            question=prompt
                        )
                        st.markdown(response)

                        # --- Charts ---
                        st.subheader("Data Visualization")

                        # Chart 1: Bed Capacity vs. Inpatient Admissions
                        df_bed_admissions = pd.DataFrame({
                            "Department": [dept for dept in selected_hospital_data["departments"]],
                            "Bed Capacity": [
                                selected_hospital_data["bed_capacity"] / len(selected_hospital_data["departments"])
                                for _ in selected_hospital_data["departments"]
                            ],
                            "Inpatient Admissions": [
                                selected_hospital_data["departments"][dept]["inpatient_admissions_daily"]
                                for dept in selected_hospital_data["departments"]
                            ]
                        })

                        fig_bed_admissions = px.bar(
                            df_bed_admissions, x="Department", y=["Bed Capacity", "Inpatient Admissions"],
                            title=f"Bed Capacity vs. Inpatient Admissions by Department in {hospital_name}",
                            barmode="group"
                        )
                        st.plotly_chart(fig_bed_admissions)

                        # Chart 2: Doctor and Nurse Ratios
                        df_staffing = pd.DataFrame({
                            "Department": [dept for dept in selected_hospital_data["departments"]],
                            "Doctors": [
                                selected_hospital_data["departments"][dept]["doctors"]
                                for dept in selected_hospital_data["departments"]
                            ],
                            "Nurses": [
                                selected_hospital_data["departments"][dept]["nurses"]
                                for dept in selected_hospital_data["departments"]
                            ]
                        })

                        fig_staffing = px.bar(
                            df_staffing, x="Department", y=["Doctors", "Nurses"],
                            title=f"Doctor and Nurse Ratios by Department in {hospital_name}",
                            barmode="group"
                        )
                        st.plotly_chart(fig_staffing)

                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            else:
                with st.chat_message("assistant"):
                    st.write(f"Could not find data for {hospital_name}.")

        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = general_chain.run(question=prompt)
                    st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Handle unclassified questions
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = general_chain.run(question=prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
