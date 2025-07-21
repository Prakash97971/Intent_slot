import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    GlobalMaxPooling1D, Dense, TimeDistributed, Dropout
)
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from data import preprocess_data, load_data

# Paths
ARTIFACT_DIR = "artifacts"
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "intent_slot_model.keras")
TOK_PATH     = os.path.join(ARTIFACT_DIR, "tokenizer.pkl")
LBL_PATH     = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")
SLOT_PATH    = os.path.join(ARTIFACT_DIR, "slot_encoder.pkl")

# Hyperparameters
MAX_LEN    = 50
VOCAB_SIZE = 8000
EMB_DIM    = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS     = 20  # Can go higher with regularization now

class ValF1Callback(Callback):
    def __init__(self, X_val, y_val_slots):
        super().__init__()
        self.X_val = X_val
        self.y_val_slots = y_val_slots

    def on_epoch_end(self, epoch, logs=None):
        pred_slots = self.model.predict(self.X_val)[1]
        pred_classes = np.argmax(pred_slots, axis=-1)
        true_classes = self.y_val_slots

        mask = true_classes != 0  # ignore 'O' pad tag
        f1 = f1_score(true_classes[mask], pred_classes[mask], average='micro')
        print(f"\n Val Slot F1: {f1:.4f}")

def build_model(vocab_size, emb_dim, lstm_units, num_intents, num_slots):
    tokens = Input(shape=(MAX_LEN,), name="tokens")
    x = Embedding(vocab_size, emb_dim, mask_zero=True)(tokens)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.3)(x)  # Regularization

    # Intent head
    pooled     = GlobalMaxPooling1D()(x)
    intent_out = Dense(num_intents, activation="softmax", name="intent")(pooled)

    # Slot head
    slot_out = TimeDistributed(
        Dense(num_slots, activation="softmax"), name="slots"
    )(x)

    model = Model(tokens, [intent_out, slot_out])
    model.compile(
        optimizer=optimizers.Adam(),
        loss={
            "intent": "sparse_categorical_crossentropy",
            "slots":  "sparse_categorical_crossentropy"
        },
        metrics={"intent": "accuracy"}
    )
    return model

def train_and_save():
    # 1) Load & preprocess
    train_data, test_data = load_data()
    (X_tr, y_tr_int, y_tr_slots), _ = preprocess_data(train_data, test_data)

    # 2) Intents: load full encoder
    intent_enc  = pickle.load(open(LBL_PATH, "rb"))
    num_intents = len(intent_enc.classes_)

    # 3) Slots: code already pads using the 'O' tag, so just count classes
    slot_enc   = pickle.load(open(SLOT_PATH, "rb"))
    num_slots  = len(slot_enc.classes_)

    print("→ intent classes/output dim:", num_intents)
    print("→ slot classes/output dim:",  num_slots)

    # 4) Split for validation
    val_split = int(0.9 * len(X_tr))
    X_train, X_val = X_tr[:val_split], X_tr[val_split:]
    y_int_train, y_int_val = y_tr_int[:val_split], y_tr_int[val_split:]
    y_slots_train, y_slots_val = y_tr_slots[:val_split], y_tr_slots[val_split:]

    # 5) Build model
    model = build_model(VOCAB_SIZE, EMB_DIM, LSTM_UNITS,
                        num_intents, num_slots)

    # 6) Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        ValF1Callback(X_val, y_slots_val)
    ]

    # 7) Train
    model.fit(
        X_train,
        {"intent": y_int_train, "slots": y_slots_train},
        validation_data=(X_val, {"intent": y_int_val, "slots": y_slots_val}),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_artifacts():
    tokenizer  = pickle.load(open(TOK_PATH, "rb"))
    intent_enc = pickle.load(open(LBL_PATH, "rb"))
    slot_enc   = pickle.load(open(SLOT_PATH, "rb"))
    model      = load_model(MODEL_PATH, compile=False)
    return tokenizer, intent_enc, slot_enc, model

def predict(sentence, tokenizer, intent_enc, slot_enc, model):
    seq    = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pi, ps = model.predict(padded)
    intent = intent_enc.inverse_transform([np.argmax(pi)])[0]
    slot_ids = np.argmax(ps[0], axis=-1)
    slot_labels = slot_enc.inverse_transform(slot_ids)
    tokens = sentence.split()
    return intent, list(zip(tokens, slot_labels[:len(tokens)]))

if __name__ == "__main__":
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    train_and_save()
