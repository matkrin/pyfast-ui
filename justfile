default:
    just --list

run:
    uv run python src/pyfast_ui/main.py

test:
    uv run pytest

build:
    uv run pyinstaller \
        --onefile \
        --name PyFastSPM \
        src/pyfast_ui/main.py

clean:
    rm -r build/ dist/ *.spec

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build
