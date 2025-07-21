# import os
# import streamlit as st
# from model import train_and_save, load_artifacts, predict

# st.set_page_config(page_title="Intent & Slot Demo", layout="wide")
# st.title("📡 Intent & Slot Detection")
# st.markdown("Enter a sentence to see predicted intent and slots.")

# MODEL_PATH = os.path.join("artifacts", "intent_slot_model.keras")
# if not os.path.exists(MODEL_PATH):
#     with st.spinner("Training model, please wait..."):
#         train_and_save()
#     st.success("Model trained successfully!")

# tokenizer, intent_enc, slot_enc, model = load_artifacts()

# user_input = st.text_input("Your sentence here:")
# if user_input:
#     intent, tags = predict(user_input, tokenizer, intent_enc, slot_enc, model)
#     st.subheader(f"**Predicted Intent:** {intent}")
#     st.table({"Word": [w for w, _ in tags], "Slot": [s for _, s in tags]})

#     html = "".join([
#         f"<span style='background-color:{('#8ecae6' if slot!='O' else '#edf2f4')};"
#         f"padding:4px;margin:2px;border-radius:4px;'>{word}<br>"
#         f"<sub>{slot}</sub></span>"
#         for word, slot in tags
#     ])
#     st.markdown(html, unsafe_allow_html=True)



import os
import streamlit as st
from model import train_and_save, load_artifacts, predict

st.set_page_config(page_title="Intent & Slot Demo", layout="centered")
st.title("✈️ Intent & Slot Detection")

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

    st.markdown("### 🧠 Predicted Intent:")
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

    st.markdown("### 🎯 Extracted Details")
    display_info("From", from_loc)
    display_info("To", to_loc)
    display_info("Departure Time", depart_time)
    display_info("Departure Date", depart_date)
    display_info("Arrival Time", arrive_time)

    st.markdown("---")
    with st.expander("📋 View Raw Slot Predictions"):
        st.dataframe(
            {"Word": [w for w, s in slots], "Slot": [s for w, s in slots]},
            use_container_width=True
        )
