import ollama
import os

_logged_prompts: set[str] = set()

def call_ollama(model: str, prompt: str, timeout_secs: int,
                num_predict: int, images: list[str] | None = None,
                system_prompt: str | None = None,
                temperature: float | None = None,
                seed: int | None = None,
                num_ctx: int | None = None) -> tuple[str, str]:
    """Einheitlicher Ollama-Call für QWEN und PHI-4-MINI.

    Args:
        prompt: Inhalt der user-Message. Bei Phase 1 nur eine kurze Aufforderung —
                das Bild hängt über `images` an dieser user-Message.
        system_prompt: Optionale system-Message (überschreibt das Modelfile-SYSTEM).
                       Bei Phase 1 wird hier der Excel-Prompt übergeben.

    Returns:
        (response_text, status) — status ist "ok", "timeout" oder "error: ..."
    """
    # Prompt-Kombination beim ersten Auftreten in Datei loggen
    log_key = f"{system_prompt}||{prompt}"
    if log_key not in _logged_prompts:
        _logged_prompts.add(log_key)
        log_path = os.path.join(os.path.dirname(__file__), "prompt_debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nMODEL: {model}\n"
                    f"SYSTEM:\n{system_prompt}\nUSER:\n{prompt}\n{'='*60}\n")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_msg = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images
    messages.append(user_msg)

    try:
        client = ollama.Client(timeout=timeout_secs)
        resp = client.chat(
            model=model,
            messages=messages,
            options={k: v for k, v in {"num_predict": num_predict,
                                       "temperature": temperature,
                                       "seed": seed,
                                       "num_ctx": num_ctx}.items() if v is not None},
        )
        return resp["message"]["content"], "ok"
    except Exception as e:
        err = str(e).lower()
        status = "timeout" if ("timed out" in err or "timeout" in err) else f"error: {e}"
        return "", status
