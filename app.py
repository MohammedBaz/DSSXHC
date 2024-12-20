import json
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Load Data and Initialize LLM ---
with open("hospital_data.json", "r") as f:
    hospital_data = json.load(f)

llm = OpenAI(openai_api_key=st.secrets["OpenAIKey"], temperature=0.2)

# --- Define Prompt Template ---
prompt_template = """
You are a hospital management consultant. Analyze the following data for {hospital_name}:

{hospital_data_str}

User's Question: {question}

Based on this data, provide a concise recommendation. Consider factors like occupancy rates, staffing levels, and length of stay.
Explain your reasoning.

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

## Charts (Optional):

*   If the data supports it, mention any charts that would be helpful to visualize the analysis and recommendations.
"""

# --- Create LangChain Chain ---
prompt = PromptTemplate(
    input_variables=["hospital_name", "hospital_data_str", "question"],
    template=prompt_template
)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Streamlit Interface ---
st.title("Hospital Performance Advisor")

# Sidebar for Hospital Selection and Question
hospital_names = [hospital["name"] for hospital in hospital_data["hospitals"]]
hospital_name = st.sidebar.selectbox("Select Hospital", hospital_names)

user_question = st.sidebar.text_area("Enter your question:", f"What is the average waiting time in {hospital_name} and how can it be improved?")

# --- Get LLM Response ---
if st.sidebar.button("Get Recommendation"):
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
            response = chain.run(
                hospital_name=hospital_name,
                hospital_data_str=hospital_data_str,
                question=user_question
            )

        # --- Display Results ---
        st.markdown(response) # Use markdown for better formatting

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
