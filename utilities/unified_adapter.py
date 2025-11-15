# utilities/unified_adapter.py

import os
import json
import time
import inspect
from typing import Optional, Dict, Any

from utilities.defence_utils import generate_dada_prompt, parse_model_response

try:
    from llama_cpp import Llama
except Exception as e:
    raise ImportError(
        "Failed to import llama_cpp.Llama. Make sure llama-cpp-python is installed and available."
    ) from e


# Map model name substrings -> default chat_format (when not provided in config)
DEFAULT_CHAT_FORMATS = {
    "vicuna": "vicuna",            # Vicuna 1.5
    "llama-3": "llama-3",          # Llama 3 Instruct
    "llama3": "llama-3",
    "llama-2": "llama-2",          # Llama 2 Chat
    "llama2": "llama-2",
    "mistral": "mistral-instruct", # Mistral Instruct
    "qwen": "qwen",                # Qwen Chat
}

def guess_chat_format(model_name: str) -> Optional[str]:
    name = model_name.lower()
    for key, fmt in DEFAULT_CHAT_FORMATS.items():
        if key in name:
            return fmt
    return None


class UnifiedAdapter:
    """
    Unified adapter for local Llama-compatible models (gguf via llama.cpp).
    - Reads profiles/model_config/{model_name}.json
    - Exposes: load_model(), query_bot(input, ctx, system)
    """

    def __init__(self, model_name: str, activate_dada: bool = False):
        self.model_name = model_name.lower()
        self.config: Optional[Dict[str, Any]] = None
        self.model: Optional[Llama] = None
        self.activate_dada = activate_dada

    def load_model(self) -> None:
        cfg_path = os.path.join("profiles", "model_config", f"{self.model_name}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"[UnifiedAdapter] Config not found at {cfg_path}")

        with open(cfg_path, "r") as fh:
            self.config = json.load(fh)
        print(f"[UnifiedAdapter] Config loaded: {self.config}")

        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError(f"[UnifiedAdapter] 'model_path' must be set in {cfg_path}")

        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[UnifiedAdapter] Model file not found: {model_path}")

        # Decide chat_format (prefer config, else guess from model_name)
        chat_format = self.config.get("chat_format") or guess_chat_format(self.model_name)

        kwargs = dict(
            model_path=model_path,
            n_ctx=self.config.get("context_window", 4096),
            n_threads=os.cpu_count(),
            verbose=self.config.get("verbose", False),
        )
        if self.config.get("use_metal", True):
            kwargs["use_metal"] = True
        if self.config.get("n_gpu_layers") is not None:
            kwargs["n_gpu_layers"] = self.config["n_gpu_layers"]

        # Only pass chat_format if this llama.cpp build supports it
        if "chat_format" in inspect.signature(Llama).parameters and chat_format:
            kwargs["chat_format"] = chat_format

        print(f"[UnifiedAdapter] Initialising model from '{model_path}' with kwargs: "
              f"{ {k:v for k,v in kwargs.items() if k!='model_path'} }")
        self.model = Llama(**kwargs)
        print(f"[UnifiedAdapter] Model '{self.model_name}' initialised.\n")

    # --- prompt builders ---

    def _build_messages(self, user_text: str, retrieved_context: str, system_prompt: str):
        """Build ChatML-like messages for llama.cpp chat API."""
        if self.activate_dada:
            # Treat DADA guardrails as additional system guidance, keep user turn clean.
            defence_text = generate_dada_prompt(system_prompt, retrieved_context, user_text)
            system_content = f"{system_prompt}\n\n{defence_text}".strip()
            user_content = f"Context from knowledge base:\n{retrieved_context}\n\nUser question: {user_text}"
        else:
            system_content = system_prompt
            user_content = f"Context from knowledge base:\n{retrieved_context}\n\nUser question: {user_text}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]

    def _fallback_manual_prompt(self, user_text: str, retrieved_context: str, system_prompt: str) -> str:
        """
        If create_chat_completion isn't available (older wheels), build a format that matches the model family.
        We avoid Llama-2 [INST] markers for Vicuna to prevent '/INST' artefacts.
        """
        name = self.model_name
        context_block = f"Context from knowledge base:\n{retrieved_context}\n\n"
        if self.activate_dada:
            # Put DADA guidance at the top as system text.
            defence_text = generate_dada_prompt(system_prompt, retrieved_context, user_text)
            system_block = f"System:\n{system_prompt}\n\n{defence_text}\n\n"
        else:
            system_block = f"System:\n{system_prompt}\n\n"

        if "vicuna" in name:
            # Vicuna-style turns
            return (
                f"{system_block}"
                f"USER: {context_block}User question: {user_text}\n"
                f"ASSISTANT:"
            )
        elif "mistral" in name:
            # Mistral-Instruct uses a similar chat template but this plain style is fine for completion
            return (
                f"{system_block}"
                f"[User]\n{context_block}User question: {user_text}\n\n"
                f"[Assistant]"
            )
        elif "qwen" in name:
            # Qwen chat style (simple)
            return (
                f"{system_block}"
                f"<|user|>\n{context_block}User question: {user_text}\n<|assistant|>\n"
            )
        else:
            # Llama-2/Llama-3 style as a reasonable default
            return (
                f"[INST] <<SYS>>\n{system_block.strip()}\n<</SYS>>\n"
                f"{context_block}"
                f"User question: {user_text} [/INST]"
            )

    # --- main call ---

    def query_bot(self, input_text: str, retrieved_context: str, system_prompt: str) -> Dict[str, Any]:
        if self.model is None or self.config is None:
            raise RuntimeError("[UnifiedAdapter] Model not loaded. Call load_model() first.")

        print(f"[UnifiedAdapter] Querying model '{self.model_name}' with input: {input_text!r}")
        start = time.time()

        max_tokens = self.config.get("max_tokens", 800)
        temperature = self.config.get("temperature", 0.2)

        use_chat = hasattr(self.model, "create_chat_completion")

        if use_chat:
            messages = self._build_messages(input_text, retrieved_context, system_prompt)
            resp = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print("Model call (chat) completed. Parsing output...")
            response_text = (
                resp.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
            ).strip()
        else:
            prompt = self._fallback_manual_prompt(input_text, retrieved_context, system_prompt)
            resp = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                # Avoid aggressive stop strings; let EOS end the turn.
            )
            print("Model call (completion) completed. Parsing output...")
            response_text = (resp.get("choices", [{}])[0].get("text", "")).strip()

        # Light, safe cleanup (no [/INST] splitting)
        # Remove very common echo prefixes if present.
        for bad_prefix in (
            "ASSISTANT:", "Assistant:", "assistant:", "Response:", "Answer:", "OUTPUT:",
        ):
            if response_text.startswith(bad_prefix):
                response_text = response_text[len(bad_prefix):].lstrip()

        if self.activate_dada:
            # optional normalisation
            response_text = parse_model_response(response_text)

        latency_ms = (time.time() - start) * 1000.0
        print(f"[UnifiedAdapter] Model responded in {latency_ms:.2f} ms. Preview: {response_text[:200]}...\n")

        return {
            "response": response_text,
            "tokens_used": len(input_text.split()),  # rough estimate
            "latency_ms": latency_ms,
        }
