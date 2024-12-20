import json
import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt

# Load data
with open("hospital_data.json", "r") as f:
    hospital_data = json.load(f)

# Streamlit Interface
st.title("Healthcare Information Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Helper function: Generate suggestions for improvement
def generate_improvement_suggestions(hospital):
    suggestions = []
    
    # Check staffing levels
    total_doctors = sum(dept["doctors"] for dept in hospital["departments"].values())
    total_nurses = sum(dept["nurses"] for dept in hospital["departments"].values())
    if total_doctors / hospital["bed_capacity"] < 0.05:
        suggestions.append("Increase the number of doctors to improve patient-to-doctor ratios.")
    if total_nurses / hospital["bed_capacity"] < 0.1:
        suggestions.append("Increase the number of nurses to enhance patient care.")

    # Technology and equipment
    suggestions.append("Invest in modern medical equipment to improve diagnostic and treatment capabilities.")

    # Patient satisfaction
    suggestions.append("Implement a patient feedback system to identify areas for improvement in care quality.")

    # Staff communication
    suggestions.append("Enhance communication and collaboration among staff through regular training and team-building activities.")

    return suggestions

# Helper function: Create visualizations for hospital data
def create_visualizations(hospital):
    department_names = list(hospital["departments"].keys())
    doctor_counts = [dept["doctors"] for dept in hospital["departments"].values()]
    nurse_counts = [dept["nurses"] for dept in hospital["departments"].values()]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    index = range(len(department_names))

    ax.bar(index, doctor_counts, bar_width, label="Doctors")
    ax.bar([i + bar_width for i in index], nurse_counts, bar_width, label="Nurses")

    ax.set_xlabel("Departments")
    ax.set_ylabel("Staff Count")
    ax.set_title("Staff Distribution by Department")
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(department_names, rotation=45)
    ax.legend()

    st.pyplot(fig)

# React to user input
if prompt := st.chat_input("Ask a question about hospitals, clinics, or polyclinics here:"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process location questions
    if "location" in prompt.lower():
        # Check if the prompt specifies a hospital, clinic, or polyclinic
        entity_match = re.search(r"(hospital|clinic|polyclinic)\s*(\d+)", prompt, re.IGNORECASE)
        if entity_match:
            entity_type = entity_match.group(1).lower()
            entity_number = entity_match.group(2)
            entity_name = f"{entity_type.capitalize()}{entity_number}"

            # Search the corresponding data
            selected_entity = None
            if entity_type == "hospital":
                selected_entity = next((item for item in hospital_data["hospitals"] if item["name"] == entity_name), None)
            elif entity_type == "clinic":
                selected_entity = next((item for item in hospital_data["clinics"] if item["name"] == entity_name), None)
            elif entity_type == "polyclinic":
                selected_entity = next((item for item in hospital_data["polyclinics"] if item["name"] == entity_name), None)

            # Generate response
            if selected_entity:
                location = selected_entity["location"]
                response = (
                    f"{entity_name} is located at:\n"
                    f"- Address: {location['address']}\n"
                    f"- City: {location['city']}\n"
                    f"- Region: {location['region']}\n"
                    f"- ZIP Code: {location['zip_code']}\n"
                    f"- Latitude: {location['latitude']}, Longitude: {location['longitude']}"
                )
            else:
                response = f"Sorry, I could not find the location of {entity_name} in the data."
        else:
            response = "I couldn't identify the hospital, clinic, or polyclinic you mentioned. Please specify the entity number."

        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Process improvement suggestions
    elif "improvement" in prompt.lower():
        entity_match = re.search(r"(hospital)\s*(\d+)", prompt, re.IGNORECASE)
        if entity_match:
            entity_number = entity_match.group(2)
            entity_name = f"Hospital{entity_number}"

            # Search the hospital data
            selected_hospital = next((item for item in hospital_data["hospitals"] if item["name"] == entity_name), None)

            if selected_hospital:
                # Generate suggestions and visualizations
                suggestions = generate_improvement_suggestions(selected_hospital)
                response = f"Some potential improvements for {entity_name} could include:\n" + "\n".join(f"- {s}" for s in suggestions)

                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.markdown("### Staff Distribution by Department")
                    create_visualizations(selected_hospital)

                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = f"Sorry, I could not find data for {entity_name}."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "I couldn't identify the hospital you mentioned. Please specify the hospital number."
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        # Fallback for unsupported questions
        response = "I'm currently equipped to provide location details and suggestions for improvements. Please ask accordingly."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
