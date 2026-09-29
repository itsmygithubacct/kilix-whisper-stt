.POSIX:
PYTHON ?= python3

.PHONY: all test lint sync clean

all: test lint

# The tests replace faster-whisper with a stand-in, so they run under any
# Python 3.10+ with nothing installed.
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -t . -v

lint:
	PYTHONPATH=src $(PYTHON) -m compileall -q src tests

sync:
	uv sync --locked

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
