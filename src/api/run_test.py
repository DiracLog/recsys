"""Testing entry point. Used by `uv run serve test`."""

import uvicorn


def main():
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )


if __name__ == "__main__":
    main()
