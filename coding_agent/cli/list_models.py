from __future__ import annotations


async def list_models(search: str | None = None) -> None:
    """Print available model providers and common models."""
    models = [
        ("anthropic", ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"]),
        ("openai", ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]),
        ("vertex", ["gemini-2.0-flash", "gemini-2.5-pro"]),
    ]
    for provider, model_list in models:
        for m in model_list:
            full = f"{provider}/{m}"
            if search is None or search.lower() in full.lower():
                print(full)
