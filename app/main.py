from pathlib import Path
import sys

# Support direct execution: uv run python app/main.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from app.agents.graph import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
