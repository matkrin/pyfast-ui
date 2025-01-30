default:
    just --list

run:
    uv run src/pyfast-ui/main.py

test:
    uv run pytest

build:
    uv run pyinstaller \
        --onedir \
        --name PyFastSPM \
        src/pyfast_ui/main.py

clean:
    rm -r build/ dist/ *.spec
