"""Production entry point. Used by `uv run serve`."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,  # for render free
        log_level="info",
    )


if __name__ == "__main__":
    main()
