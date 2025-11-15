# utilities/db_sync.py
from supabase import create_client
import json
import os
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Base path relative to project root
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets")

BUSINESS_DATA_DIR = os.path.join(BASE_PATH, "knowledge_base_cache")
ATTACKS_DATA_DIR = os.path.join(BASE_PATH, "attacks_cache")

os.makedirs(BUSINESS_DATA_DIR, exist_ok=True)
os.makedirs(ATTACKS_DATA_DIR, exist_ok=True)


def fetch_all_business_data():
    """Fetch all business data and save each row as a separate JSON file based on usecase"""
    response = supabase.table("business_data").select("*").execute()
    for row in response.data:
        usecase = row["usecase"]
        data = row["data"]
        file_path = os.path.join(BUSINESS_DATA_DIR, f"{usecase}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved business data for usecase '{usecase}' to {file_path}")


def fetch_all_attacks():
    """Fetch all attacks and save as a single JSON file in flat array format"""
    response = supabase.table("prompt_injection_attacks").select("*").execute()
    
    attacks_list = []
    for row in response.data:
        attack_entry = {
            "id": row.get("id"),
            "usecase": row.get("usecase"),
            "attack_name": row.get("attack_name"),
            "attack_family": row.get("attack_family"),
            "attack_prompt": row.get("attack_prompt"),
        }

        attacks_list.append(attack_entry)

    file_path = os.path.join(ATTACKS_DATA_DIR, "attacks.json")
    with open(file_path, "w") as f:
        json.dump(attacks_list, f, indent=2)
    print(f"[INFO] Saved all attacks to {file_path}")


if __name__ == "__main__":
    fetch_all_business_data()
    fetch_all_attacks()
