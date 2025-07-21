import os
import streamlit as st
from model import train_and_save, load_artifacts, predict

st.set_page_config(page_title="Intent & Slot Demo", layout="centered")
st.title("Intent & Slot Detection")

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "intent_slot_model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")

# Load model and metadata
model, tokenizer, label_encoder, slot_label_map = load_artifacts()
# User input
user_input = st.text_input("🗣️ Enter your sentence:", placeholder="e.g., Book me a flight from Delhi to Mumbai tomorrow at 9 AM")

if user_input:
    intent, slots = predict(user_input, model, tokenizer, label_encoder, slot_label_map)

    st.markdown("### Predicted Intent:")
    st.success(f"**{intent}**")

    # Group slot predictions
    from_loc = []
    to_loc = []
    depart_time = []
    arrive_time = []
    depart_date = []

    for word, slot in slots:
        if "fromloc" in slot:
            from_loc.append(word)
        elif "toloc" in slot:
            to_loc.append(word)
        elif "depart_time" in slot:
            depart_time.append(word)
        elif "arrive_time" in slot:
            arrive_time.append(word)
        elif "depart_date" in slot or "date" in slot:
            depart_date.append(word)

    def display_info(label, content):
        if content:
            st.markdown(f"**{label}**: {' '.join(content)}")

    st.markdown("### Extracted Details")
    display_info("From", from_loc)
    display_info("To", to_loc)
    display_info("Departure Time", depart_time)
    display_info("Departure Date", depart_date)
    display_info("Arrival Time", arrive_time)

    st.markdown("---")
    with st.expander("View Raw Slot Predictions"):
        st.dataframe(
            {"Word": [w for w, s in slots], "Slot": [s for w, s in slots]},
            use_container_width=True
        )
