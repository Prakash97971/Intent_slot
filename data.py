# data.py
import os
import pickle
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

# Configuration
DATASET_NAME = "tuetschek/atis"
ARTIFACT_DIR = "artifacts"
TOK_PATH     = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")
LBL_PATH     = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")
SLOT_PATH    = os.path.join(ARTIFACT_DIR, "slot_encoder.pkl")
MAX_LEN      = 50
VOCAB_SIZE   = 8000

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_data():
    ds = load_dataset(DATASET_NAME)
    full = ds["train"]
    splits = full.train_test_split(test_size=0.2, seed=42)
    return splits["train"], splits["test"]

def preprocess_data(train, test):
    # Texts & intents
    texts_tr, intents_tr = zip(*[(ex["text"], ex["intent"]) for ex in train])
    texts_te, intents_te = zip(*[(ex["text"], ex["intent"]) for ex in test])

    # Slots lists
    slots_tr = [ex["slots"].split() for ex in train]
    slots_te = [ex["slots"].split() for ex in test]

    # 1) Tokenizer on texts
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_SIZE, oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts_tr)
    with open(TOK_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    seq_tr = tokenizer.texts_to_sequences(texts_tr)
    seq_te = tokenizer.texts_to_sequences(texts_te)
    X_tr   = pad_sequences(seq_tr, maxlen=MAX_LEN, padding="post")
    X_te   = pad_sequences(seq_te, maxlen=MAX_LEN, padding="post")

    # 2) Intent encoding
    intent_enc = LabelEncoder()
    intent_enc.fit(list(intents_tr) + list(intents_te))
    with open(LBL_PATH, "wb") as f:
        pickle.dump(intent_enc, f)
    y_tr_int = intent_enc.transform(intents_tr)
    y_te_int = intent_enc.transform(intents_te)

    # 3) Slot encoding
    all_slots = [s for seq in slots_tr for s in seq] + [s for seq in slots_te for s in seq]
    slot_enc  = LabelEncoder()
    slot_enc.fit(all_slots)
    with open(SLOT_PATH, "wb") as f:
        pickle.dump(slot_enc, f)

    # Use the index of the 'O' tag as our pad value
    pad_label = slot_enc.transform(["O"])[0]

    # Transform & pad with that exact index
    y_tr_slots = pad_sequences(
        [slot_enc.transform(seq) for seq in slots_tr],
        maxlen=MAX_LEN, padding="post", value=pad_label
    )
    y_te_slots = pad_sequences(
        [slot_enc.transform(seq) for seq in slots_te],
        maxlen=MAX_LEN, padding="post", value=pad_label
    )

    return (X_tr, y_tr_int, y_tr_slots), (X_te, y_te_int, y_te_slots)

if __name__ == "__main__":
    tr, te = load_data()
    preprocess_data(tr, te)
    print(f"✅ Artifacts saved in: {os.path.abspath(ARTIFACT_DIR)}")
