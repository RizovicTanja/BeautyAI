import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_JSON = os.path.join(BASE_DIR, "..", "data", "makeup_data.json") 

DATA_JSON = os.path.normpath(DATA_JSON)

def load_products():
    """Učitaj proizvode iz makeup_data.json"""
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

