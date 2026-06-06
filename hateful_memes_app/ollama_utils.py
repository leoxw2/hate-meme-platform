import ollama

def call_ollama(model: str, prompt: str, timeout_secs: int,
                num_predict: int, images: list[str] | None = None) -> tuple[str, str]:
    """Einheitlicher Ollama-Call für QWEN und PHI-4-MINI.

    Returns:
        (response_text, status) — status ist "ok", "timeout" oder "error: ..."
    """
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images

    try:
        client = ollama.Client(timeout=timeout_secs)
        resp = client.chat(
            model=model,
            messages=[msg],
            options={"num_predict": num_predict},
        )
        return resp["message"]["content"], "ok"
    except Exception as e:
        err = str(e).lower()
        status = "timeout" if ("timed out" in err or "timeout" in err) else f"error: {e}"
        return "", status
