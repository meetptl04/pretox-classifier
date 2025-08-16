import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Streamlit UI
st.title("BioBERT Text Classifier")
st.write("This app classifies biomedical text into **PRETOX_REL** or **NO_PRETOX_REL**.")

st.subheader("Try these example texts:")
examples = [
    "The compound caused liver toxicity in preclinical studies.",
    "The drug showed no adverse effects in animal trials.",
    "Toxicological assessment revealed potential kidney damage.",
]
for i, ex in enumerate(examples, 1):
    if st.button(f"Example {i}: {ex[:50]}..."):
        st.session_state["user_input"] = ex

user_input = st.text_area("Enter your text:", value=st.session_state.get("user_input", ""))

if st.button("Predict"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "NO_PRETOX_REL", 1: "PRETOX_REL"}
    st.success(f"Prediction: **{label_map[pred]}**")
