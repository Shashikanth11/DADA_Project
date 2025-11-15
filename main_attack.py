# main_attack.py

import argparse
import json
import os
from typing import List
import sys
from socket import timeout as TimeoutError # Ensure TimeoutError is accessible

from utilities.rag_utils import rag_retrieve
from utilities.evaluation import is_attack_success  # returns bool
from utilities.unified_adapter import UnifiedAdapter

# Directories
USECASE_DIR = os.path.join("profiles", "usecase_config")
MODEL_CONFIG_DIR = os.path.join("profiles", "model_config")
ATTACKS_CACHE_PATH = os.path.join("datasets", "attacks_cache", "attacks.json")
RESULTS_DIR = "results"


def load_usecase_config(usecase: str) -> dict:
    """Load usecase config JSON"""
    config_path = os.path.join(USECASE_DIR, f"{usecase}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[main_attack] Usecase config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_cached_attacks() -> list:
    """Load cached attacks JSON"""
    if not os.path.exists(ATTACKS_CACHE_PATH):
        raise FileNotFoundError(f"[main_attack] Attacks cache not found: {ATTACKS_CACHE_PATH}")
    with open(ATTACKS_CACHE_PATH, "r") as f:
        return json.load(f)


def save_results(model: str, usecase: str, results: list, activate_dada: bool = False) -> None:
    """Save results under results/{model}_{usecase}.json"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Add suffix if defense is active
    defense_suffix = "_defended" if activate_dada else ""
    filename = f"{model}_{usecase}{defense_suffix}.json"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {path}")

#1----------------------------------------------
def discover_available_models() -> List[str]:
    """Discover model names by enumerating JSON files in profiles/model_config"""
    if not os.path.exists(MODEL_CONFIG_DIR):
        return []
    files = [
        os.path.splitext(f)[0]
        for f in os.listdir(MODEL_CONFIG_DIR)
        if f.endswith(".json")
    ]
    return [f.lower() for f in files]
#------------------------------------------------
def run_attack(model: str, usecase: str, activate_bully: bool = False, activate_dada: bool = False) -> None:
    """Run attacks for a given model-usecase combination"""
    if activate_bully:
        print(f"[STUB] Bully agent not implemented. Skipping adversarial mode for {model}-{usecase}.")
        return

    print(f"[INFO] Running attacks on model={model}, usecase={usecase}")

    # --------------------------
    # Initialise unified adapter for the requested model
    adapter = UnifiedAdapter(model,activate_dada=activate_dada)
    try:
        adapter.load_model()
    except Exception as e:
        print(f"[ERROR] Failed to initialise model '{model}': {e}")
        return

    # --------------------------
    # Load usecase config once
    try:
        usecase_config = load_usecase_config(usecase)
    except FileNotFoundError as e:
        print(f"[WARN] {e}. Skipping usecase '{usecase}'.")
        return

    # --------------------------
    # Load cached attacks once
    try:
        all_attacks = load_cached_attacks()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}. Cannot proceed without attacks cache.")
        return

    attacks = [
        atk for atk in all_attacks
        if atk.get("usecase", "").lower() in [usecase.lower(), "general"]
    ]

    # ✅ NEW: Load FAISS index and embedding model ONCE per usecase
    print(f"[INFO] Loading FAISS index and embedding model for {usecase}...")
    from utilities.rag_utils import load_index_and_docs, get_embedder
    index, docs = load_index_and_docs(usecase)
    embedder = get_embedder()
    print(f"[INFO] Loaded {len(docs)} docs for {usecase} RAG context.")

    results = []

    for atk in attacks:
        attack_name = atk.get("attack_name", "unknown")
        attack_family = atk.get("attack_family", "unknown")
        attack_prompt = atk.get("attack_prompt", "")

        # ✅ Pass preloaded objects to rag_retrieve
        context = rag_retrieve(
            attack_prompt,
            usecase,
            top_k=usecase_config.get("top_k", 3),
            embedder=embedder,
            index=index,
            docs=docs
        )
        print(f"[DEBUG] Attack prompt: {attack_prompt}")

        response = adapter.query_bot(
            attack_prompt,
            context,
            system_prompt=usecase_config.get("system_prompt", "")
        )
        print(f"[DEBUG] Model response: {response}")

        # --- Start Evaluation Phase with Timeout Handling ---
        attack_success = 'evaluation_error' # Default error state before evaluation attempt

        try:
            # NOTE: The is_attack_success function is where the judge call happens
            attack_success = is_attack_success(
                model_response=response["response"],
                attack_prompt=attack_prompt,
                usecase=usecase,
                system_prompt=usecase_config.get("system_prompt", "")
            )

        except TimeoutError:
            # This catches the specific socket timeout (300s) from the judge model call
            attack_success = 'timeout'
            print(f"[ERROR] Judge evaluation timed out for attack: '{attack_prompt[:50]}...'")
            # Optional: Log the full stack trace for detailed debugging
            # print(f"Full Error Trace:\n{sys.exc_info()}")

        except Exception as e:
            # Catch any other unexpected errors during evaluation (e.g., connection errors, JSON parsing issues)
            attack_success = 'judge_failure'
            print(f"[ERROR] Judge evaluation failed with exception: {e}")
        # --- End Evaluation Phase with Timeout Handling ---


        results.append({
            "attack_family": attack_family,
            "attack_name": attack_name,
            "attack_prompt": attack_prompt,
            "model_response": response["response"],
            "attack_success": attack_success,
            "latency_ms": response.get("latency_ms"),
            "tokens_used": response.get("tokens_used"),
            "defence_active": activate_dada,   # log whether defence was active
        })

        print(f"  - {attack_family} | success={attack_success}")

    save_results(model, usecase, results,activate_dada=activate_dada)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name, e.g., mistral or 'all'")
    parser.add_argument("--usecase", required=True, help="Usecase name, e.g., rental or 'all'")
    parser.add_argument("--activatebully", action="store_true", help="Enable bully adversarial agent mode")
    parser.add_argument("--activate_dada", action="store_true", help="Activate DADA defence mechanism")

    args = parser.parse_args()
    
    #4----------------------------------------------
    requested_model = args.model.lower()
    requested_usecase = args.usecase.lower()

    # Expand models
    if requested_model == "all":
        models = discover_available_models()
        if not models:
            print("[WARN] No model configs found under profiles/model_config/. Exiting.")
            return
    else:
        models = [requested_model]

    # Expand usecases
    if requested_usecase == "all":
        if not os.path.exists(USECASE_DIR):
            print(f"[WARN] Usecase directory not found: {USECASE_DIR}. Exiting.")
            return
        usecases = [
            os.path.splitext(f)[0].lower()
            for f in os.listdir(USECASE_DIR) if f.endswith(".json")
        ]
    else:
        usecases = [requested_usecase]

    print(f"[INFO] Selected models={models}, usecases={usecases}")
    #------------------------------------------------

    for model in models:
        for usecase in usecases:
            run_attack(
                model,
                usecase,
                activate_bully=args.activatebully,
                activate_dada=args.activate_dada
            )


if __name__ == "__main__":
    main()
