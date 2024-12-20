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

    # Check if the question is about waiting time or a specific hospital
    if "waiting time" in prompt.lower():
        # Calculate and display average waiting time
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
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": f"The average waiting time across all healthcare centers in Taif is approximately {avg_wait_time_all:.2f} days."})

            # Check if the question is about the hospital with the highest waiting time
            if "highest waiting time" in prompt.lower():
                highest_waiting_time_hospital = max(hospital_waiting_times, key=hospital_waiting_times.get)
                highest_waiting_time = hospital_waiting_times[highest_waiting_time_hospital]
                with st.chat_message("assistant"):
                    st.write(f"The hospital with the highest waiting time is {highest_waiting_time_hospital} with an average waiting time of approximately {highest_waiting_time:.2f} days.")
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": f"The hospital with the highest waiting time is {highest_waiting_time_hospital} with an average waiting time of approximately {highest_waiting_time:.2f} days."})

        else:
            with st.chat_message("assistant"):
                st.write("Could not calculate average waiting time due to lack of data.")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": "Could not calculate average waiting time due to lack of data."})

    else:
        # Check if the question is about a specific hospital
        match = re.search(r"hospital\s*(\w+)", prompt, re.IGNORECASE)  # Improved regex
        if match:
            hospital_name = f"Hospital{match.group(1)}"

            # Check if it's a simple information request
            if prompt.lower().startswith("where is") or prompt.lower().startswith("what is the location of"):
                # Find the hospital
                selected_hospital_data = None
                for hospital in hospital_data["hospitals"]:
                    if hospital["name"] == hospital_name:
                        selected_hospital_data = hospital
                        break
                if selected_hospital_data:
                    location = selected_hospital_data["location"]
                    with st.chat_message("assistant"):
                        st.write(f"{hospital_name} is located in {location['city']}, {location['region']}, at {location['address']}.")
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": f"{hospital_name} is located in {location['city']}, {location['region']}, at {location['address']}."})
                else:
                    with st.chat_message("assistant"):
                        st.write(f"Could not find information for {hospital_name}.")
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": f"Could not find information for {hospital_name}."})

            else:
                hospital_data_str = json.dumps(hospital_data, indent=2)
                # Find the selected hospital's data
                selected_hospital_data = None
                for hospital in hospital_data["hospitals"]:
                    if hospital["name"] == hospital_name:
                        selected_hospital_data = hospital
                        break

                # Display assistant response in chat message container only if hospital is found
                if selected_hospital_data:
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

        else:
            # Use the general chain for general questions
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = general_chain.run(question=prompt)
                    st.write(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
