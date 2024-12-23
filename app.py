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
You are a helpful AI assistant. Answer the following question as concisely as possible.

Here are some examples:

Question: What are the major causes of heart disease?
Answer: The major causes of heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, unhealthy diet, physical inactivity, and excessive alcohol use.

Question: How can I improve my hospital's efficiency?
Answer: You can improve your hospital's efficiency by implementing lean management principles, improving staff scheduling, optimizing patient flow, and investing in technology such as electronic health record (EHR) systems.

Question: What is the best way to manage hospital staff schedules?
Answer: The best way to manage hospital staff schedules is to use scheduling software that takes into account staff availability, patient volume, and regulatory requirements.

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

    Here are some examples:

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

    Question: What is the average waiting time of Hospital1?
    Answer:
    ## Analysis:
    *   Hospital1 has a total bed capacity of 400.
    *   The daily inpatient admissions across all departments is 110.
    *   The average waiting time is calculated as bed capacity divided by total daily inpatient admissions.

    ## Considerations:
    *  This calculation assumes uniform distribution of patients across all departments and does not account for variations in patient flow or emergency cases.

    ## Conclusion:

    *   The average waiting time for Hospital1 is approximately 3.64 days based on the provided data and calculation method.

    ## Recommendations:

    1.  Further investigate the distribution of patient admissions and lengths of stay within each department to identify bottlenecks.
    2.  Consider implementing a patient flow management system to optimize bed usage and reduce waiting times.

    Question: what are the location and bed capacity of Hospital1?
    Answer:
    ## Analysis:

    *   Hospital1 is located in Taif, Makkah, at Al Mathnah, Taif.
    *   Hospital1 has a bed capacity of 400.

    ## Considerations:

    *   The bed capacity indicates the maximum number of inpatients the hospital can accommodate.

    ## Conclusion:

    *   Hospital1 is situated at Al Mathnah in Taif, Makkah, and can accommodate up to 400 inpatients.

    ## Recommendations:

    1.  Regularly assess the occupancy rates to ensure the hospital is operating efficiently within its capacity.
    2.  Consider expansion or resource reallocation if the hospital frequently operates near or at full capacity.

    Question: what is the average waiting time of Hospital2?
    Answer:
    ## Analysis:
    *   Hospital2 has a total bed capacity of 250.
    *   The daily inpatient admissions across all departments is 65.
    *   The average waiting time is calculated as bed capacity divided by total daily inpatient admissions.

    ## Considerations:
    *  This calculation assumes uniform distribution of patients across all departments and does not account for variations in patient flow or emergency cases.

    ## Conclusion:

    *   The average waiting time for Hospital2 is approximately 3.85 days based on the provided data and calculation method.

    ## Recommendations:

    1.  Further investigate the distribution of patient admissions and lengths of stay within each department to identify bottlenecks.
    2.  Consider implementing a patient flow management system to optimize bed usage and reduce waiting times.

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

    # Check for specific keywords or patterns
    waiting_time_keywords = ["waiting time", "wait time", "how long to wait"]
    hospital_specific_pattern = r"hospital\s*(\w+)"

    hospital_match = re.search(hospital_specific_pattern, prompt, re.IGNORECASE)

    if any(keyword in prompt.lower() for keyword in waiting_time_keywords):
        # Handle waiting time questions
        if hospital_match:
            # Handle waiting time for a specific hospital
            hospital_name = f"Hospital{hospital_match.group(1)}"
            selected_hospital_data = next((hospital for hospital in hospital_data["hospitals"] if hospital["name"] == hospital_name), None)

            if selected_hospital_data:
                total_beds = selected_hospital_data["bed_capacity"]
                total_admissions = sum(selected_hospital_data["departments"][dept]["inpatient_admissions_daily"] for dept in selected_hospital_data["departments"])
                if total_admissions > 0:
                    avg_wait_time = total_beds / total_admissions
                    with st.chat_message("assistant"):
                        st.write(f"The average waiting time in {hospital_name} is approximately {avg_wait_time:.2f} days.")
                    st.session_state.messages.append({"role": "assistant", "content": f"The average waiting time in {hospital_name} is approximately {avg_wait_time:.2f} days."})
                else:
                    with st.chat_message("assistant"):
                        st.write(f"Could not calculate average waiting time for {hospital_name} due to lack of data.")
                    st.session_state.messages.append({"role": "assistant", "content": f"Could not calculate average waiting time for {hospital_name} due to lack of data."})
            else:
                with st.chat_message("assistant"):
                    st.write(f"Could not find data for {hospital_name}.")
                st.session_state.messages.append({"role": "assistant", "content": f"Could not find data for {hospital_name}."})
        else:
            # Handle general waiting time questions
            wait_times = []
            hospital_waiting_times = {}
            for hospital in hospital_data["hospitals"]:
                total_beds = hospital["bed_capacity"]
                total_admissions = sum(hospital["departments"][dept]["inpatient_admissions_daily"] for dept in hospital["departments"])
                if total_admissions > 0:
                    avg_wait_time = total_beds / total_admissions
                    wait_times.append(avg_wait_time)
                    hospital_waiting_times[hospital["name"]] = avg_wait_time

            if wait_times:
                avg_wait_time_all = sum(wait_times) / len(wait_times)
                with st.chat_message("assistant"):
                    st.write(f"The average waiting time across all healthcare centers in Taif is approximately {avg_wait_time_all:.2f} days.")
                st.session_state.messages.append({"role": "assistant", "content": f"The average waiting time across all healthcare centers in Taif is approximately {avg_wait_time_all:.2f} days."})

                if "highest waiting time" in prompt.lower():
                    highest_waiting_time_hospital = max(hospital_waiting_times, key=hospital_waiting_times.get)
                    highest_waiting_time = hospital_waiting_times[highest_waiting_time_hospital]
                    with st.chat_message("assistant"):
                        st.write(f"The hospital with the highest waiting time is {highest_waiting_time_hospital} with an average waiting time of approximately {highest_waiting_time:.2f} days.")
                    st.session_state.messages.append({"role": "assistant", "content": f"The hospital with the highest waiting time is {highest_waiting_time_hospital} with an average waiting time of approximately {highest_waiting_time:.2f} days."})

            else:
                with st.chat_message("assistant"):
                    st.write("Could not calculate average waiting time due to lack of data.")
                st.session_state.messages.append({"role": "assistant", "content": "Could not calculate average waiting time due to lack of data."})

    elif hospital_match:
        # Handle hospital-specific questions
        hospital_name = f"Hospital{hospital_match.group(1)}"
        hospital_data_str = json.dumps(hospital_data, indent=2)

        # Check if it's a request to list all hospital names
        if prompt.lower() == "list the name of all hospitals":
            with st.chat_message("assistant"):
                hospital_names = [hospital["name"] for hospital in hospital_data["hospitals"]]
                st.write("List of all hospitals:")
                for name in hospital_names:
                    st.write(name)
                st.session_state.messages.append({"role": "assistant", "content": "\n".join(hospital_names)})

        # Find the selected hospital's data
        selected_hospital_data = None
        for hospital in hospital_data["hospitals"]:
            if hospital["name"] == hospital_name:
                selected_hospital_data = hospital
                break

        # Check if the question is a simple location query
        if prompt.lower().startswith("where is the location of"):
            if selected_hospital_data:
                location = selected_hospital_data["location"]
                with st.chat_message("assistant"):
                    st.write(f"{hospital_name} is located in {location['city']}, {location['region']}, at {location['address']}.")
                st.session_state.messages.append({"role": "assistant", "content": f"{hospital_name} is located in {location['city']}, {location['region']}, at {location['address']}."})
            else:
                with st.chat_message("assistant"):
                    st.write(f"Could not find data for {hospital_name}.")
                st.session_state.messages.append({"role": "assistant", "content": f"Could not find data for {hospital_name}."})

        # Handle other hospital-specific questions using analysis_chain
        elif selected_hospital_data:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data and generating recommendation..."):
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
                        "Bed Capacity": [selected_hospital_data["bed_capacity"] / len(selected_hospital_data["departments"]) for _ in selected_hospital_data["departments"]],
                        "Inpatient Admissions": [selected_hospital_data["departments"][dept]["inpatient_admissions_daily"] for dept in selected_hospital_data["departments"]]
                    })

                    fig_bed_admissions = px.bar(df_bed_admissions, x="Department", y=["Bed Capacity", "Inpatient Admissions"],
                                                 title=f"Bed Capacity vs. Inpatient Admissions by Department in {hospital_name}",
                                                 barmode="group")
                    st.plotly_chart(fig_bed_admissions)

                    # Chart 2: Doctor and Nurse Ratios
                    df_staffing = pd.DataFrame({
                        "Department": [dept for dept in selected_hospital_data["departments"]],
                        "Doctors": [selected_hospital_data["departments"][dept]["doctors"] for dept in selected_hospital_data["departments"]],
                        "Nurses": [selected_hospital_data["departments"][dept]["nurses"] for dept in selected_hospital_data["departments"]]
                    })

                    fig_staffing = px.bar(df_staffing, x="Department", y=["Doctors", "Nurses"],
                                           title=f"Doctor and Nurse Ratios by Department in {hospital_name}",
                                           barmode="group")
                    st.plotly_chart(fig_staffing)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            with st.chat_message("assistant"):
                st.write(f"Could not find data for {hospital_name}.")
            st.session_state.messages.append({"role": "assistant", "content": f"Could not find data for {hospital_name}."})

    else:
        # Handle general questions
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = general_chain.run(question=prompt)
                st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
