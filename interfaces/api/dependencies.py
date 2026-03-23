from fastapi import HTTPException, Request


def get_model_loader(request: Request):
    """Returns the loaded ModelLoader or 503 if models failed to load (e.g. missing HF_TOKEN)."""
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODELS_NOT_LOADED",
                "message": "Classification models are not loaded. Set HF_TOKEN in .env and restart.",
            },
        )
    return loader
