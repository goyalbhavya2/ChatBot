import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from safetensors.torch import load_file
import torch
import os
model_dir = "."
#safetensors_file = os.path.join(model_dir, "adapter_model.safetensors")
#pytorch_model_file = os.path.join(model_dir, "pytorch_model.bin")
#state_dict = load_file(safetensors_file)
#torch.save(state_dict, pytorch_model_file)


@st.cache_resource
def load_model_and_tokenizer(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_dir)
def generate_response(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt",max_length=128).input_ids
    outputs = model.generate(input_ids, max_length=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("Email Generation Chatbot")

subjects = {
    "Request for Leave": [
        "Vacation Request",
        "Request for Personal Leave",
        "Annual Leave Request",
        "Medical Request Leave"
    ],
    "Status Report": [
        "Weekly Status Update",
        "Project Status Report",
        "Milestone Achievement Report",
        "Progress Overview"
    ],
    "Performance Review Request": [
        "Performance Review Meeting Request",
        "Performance Evaluation Discussion Request"
    ],
    "Salary Increment Request": [
        "Request for Salary Increase Review"
    ],
    "Resource Allocation Request": [
        "Request for Additional Resources",
        "Resource Allocation Inquiry",
        "Changes in Resource Allocation"
    ]
}

st.subheader("Available Subjects for Email Generation:")
for category, sublist in subjects.items():
    st.write(f"**{category}**")
    for subject in sublist:
        st.write(f"- {subject}")

input_text = st.text_area("Enter the subject and details for the email:", "")

if st.button("Generate Email"):
    if input_text:
        generated_email = generate_response(input_text)
        st.write("### Generated Email:")
        st.write(generated_email)
    else:
        st.warning("Please enter the subject and details for the email.")
