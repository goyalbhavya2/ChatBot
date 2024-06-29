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


def load_model_and_tokenizer(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_dir)
def generate_response(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("Email Generation Chatbot")

input_text = st.text_area("Enter your request for an email:", "")
if st.button("Generate Email"):
    generated_email = generate_response(input_text)
    st.write(generated_email)
