
import pickle
import os

def save_model(model, filename):
    os.makedirs("../models", exist_ok=True)
    with open(f"../models/{filename}", "wb") as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(f"../models/{filename}", "rb") as f:
        return pickle.load(f)
