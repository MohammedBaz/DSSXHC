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

llm = OpenAI(openai_api_key=st.secrets["OpenAIKey"], temperature=0.2)  # Use for general questions
chat_llm = ChatOpenAI(openai_api_key=st.secrets["OpenAIKey"], temperature=0.2) # Use for data-specific analysis

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

# Hospital Analysis Prompt Template
system_message_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are a hospital management consultant. Analyze the following data:
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
    
    Example of a question and answer:
    
    Question: Do you think it would be better to increase the bed capacity of hospital x to 100?
    
    Answer:
    ## Analysis:
    * The current bed capacity of Hospital X is 80.
    * The overall occupancy rate is 79%.
    * The surgery department has the highest occupancy rate at 90%, with an average stay of 2 days.
    * The surgery department has 5 doctors and 50 nurses.

    ## Considerations:

    * Increasing bed capacity without addressing the doctor shortage in the surgery department might not be effective.
    * The high occupancy rate in surgery suggests a potential bottleneck.
    * The short average stay in surgery indicates a high turnover of patients.

    ## Conclusion:

    Increasing the bed capacity to 100 might not be the most effective solution without addressing the staffing issue in the surgery department.

    ## Recommendations:

    1.  Prioritize increasing the number of doctors in the surgery department.
    2.  Monitor occupancy rates after increasing the number of doctors to determine if further bed capacity expansion is needed.

    ## Further Considerations (Optional):

    * Analyze patient wait times in the surgery department.
    * Evaluate the efficiency of the surgical scheduling process.
    
    """
)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

analysis_chain = LLMChain(llm=chat_llm, prompt=chat_prompt)

# --- Streamlit Interface ---
st.title("Healthcare Advisor")

# Sidebar for Hospital Selection and Question
hospital_names = [hospital["name"] for hospital in hospital_data["hospitals"]]
hospital_names.insert(0, "All Healthcare Centers")  # Add an option for all centers
hospital_name = st.sidebar.selectbox("Select Healthcare Center", hospital_names)

user_question = st.sidebar.text_area("Enter your question:", "What is the average waiting time in all healthcare centers in Taif?")

# --- Get LLM Response ---
if st.sidebar.button("Get Answer"):
    if hospital_name == "All Healthcare Centers":
        # Check if the question is about a specific hospital
        match = re.search(r"hospital\s*(\d+)", user_question, re.IGNORECASE)
        if match:
            hospital_number = match.group(1)
            hospital_name = f"Hospital{hospital_number}"

            selected_hospital_data = None
            for hospital in hospital_data["hospitals"]:
                if hospital["name"] == hospital_name:
                    selected_hospital_data = hospital
                    break

            if selected_hospital_data:
                hospital_data_str = json.dumps(selected_hospital_data, indent=2)

                with st.spinner("Analyzing data and generating recommendation..."):
                    response = analysis_chain.run(
                        hospital_name=hospital_name,
                        hospital_data_str=hospital_data_str,
                        question=user_question
                    )
                st.markdown(response)

            else:
                st.error(f"Data for {hospital_name} not found.")

        elif "waiting time" in user_question.lower():
            # Existing waiting time calculation for all centers
            wait_times = []
            for hospital in hospital_data["hospitals"]:
                total_beds = hospital["bed_capacity"]
                total_admissions = sum(hospital["departments"][dept]["inpatient_admissions_daily"] for dept in hospital["departments"])
                if total_admissions > 0:
                    avg_wait_time = total_beds / total_admissions
                    wait_times.append(avg_wait_time)

            if wait_times:
                avg_wait_time_all = sum(wait_times) / len(wait_times)
                st.subheader("Average Waiting Time:")
                st.write(f"The average waiting time across all healthcare centers in Taif is approximately {avg_wait_time_all:.2f} days.")
            else:
                st.write("Could not calculate average waiting time due to lack of data.")

        else:
            # Use the general chain for general questions
            with st.spinner("Thinking..."):
                response = general_chain.run(question=user_question)
            st.subheader("Answer:")
            st.write(response)
    else:
        # Existing hospital specific analysis
        selected_hospital_data = None
        for hospital in hospital_data["hospitals"]:
            if hospital["name"] == hospital_name:
                selected_hospital_data = hospital
                break

        if selected_hospital_data is None:
            st.error(f"Data for {hospital_name} not found.")
        else:
            hospital_data_str = json.dumps(selected_hospital_data, indent=2)

            with st.spinner("Analyzing data and generating recommendation..."):
                response = analysis_chain.run(
                    hospital_name=hospital_name,
                    hospital_data_str=hospital_data_str,
                    question=user_question
                )

            # --- Display Results ---
            st.markdown(response)

            # --- Charts ---
            st.subheader("Data Visualization")

            # Chart 1: Bed Capacity vs. Inpatient Admissions
            df_bed_admissions = pd.DataFrame({
                "Department": [dept for dept in selected_hospital_data["departments"]],
                "Bed Capacity": [selected_hospital_data["bed_capacity"] / len(selected_hospital_data["departments"]) for _ in selected_hospital_data["departments"]],
                "Inpatient Admissions": [selected_hospital_data["departments"][dept]["inpatient_admissions_daily"] for dept in selected_hospital_data["departments"]]
            })

            fig_bed_admissions = px.bar(df_bed_admissions, x="Department", y=["Bed Capacity", "Inpatient Admissions"],
                                         title="Bed Capacity vs. Inpatient Admissions by Department",
                                         barmode="group")
            st.plotly_chart(fig_bed_admissions)

            # Chart 2: Doctor and Nurse Ratios
            df_staffing = pd.DataFrame({
                "Department": [dept for dept in selected_hospital_data["departments"]],
                "Doctors": [selected_hospital_data["departments"][dept]["doctors"] for dept in selected_hospital_data["departments"]],
                "Nurses": [selected_hospital_data["departments"][dept]["nurses"] for dept in selected_hospital_data["departments"]]
            })

            fig_staffing = px.bar(df_staffing, x="Department", y=["Doctors", "Nurses"],
                                   title="Doctor and Nurse Ratios by Department",
                                   barmode="group")
            st.plotly_chart(fig_staffing)

            # --- Optional: Display Data ---
            if st.checkbox("Show Hospital Data"):
                st.json(hospital_data_str)
