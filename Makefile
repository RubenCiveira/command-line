VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PID_FILE := .window.pid
ENTRY    := src/window.py

.PHONY: build test lint format clean run stop restart status

## build: create virtualenv and install all dependencies (prod + dev)
build: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio ruff
	touch $(VENV)/bin/activate

## test: run the full test suite
test: build
	PYTHONPATH=src $(VENV)/bin/pytest test/ -v

## lint: static analysis with ruff
lint: build
	$(VENV)/bin/ruff check .

## format: auto-format sources with ruff
format: build
	$(VENV)/bin/ruff format .

## clean: remove virtualenv and cached files
clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

run: build
	PYTHONPATH=src $(PYTHON) $(ENTRY)

stop:
	@set -e; \
	if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat "$(PID_FILE)" 2>/dev/null || true); \
		if [ -n "$$PID" ] && kill -0 $$PID 2>/dev/null; then \
			echo "Stopping app (pid=$$PID)"; \
			kill $$PID 2>/dev/null || true; \
		else \
			echo "No running process for pid=$$PID"; \
		fi; \
		rm -f "$(PID_FILE)"; \
	else \
		echo "No PID file"; \
	fi

restart: stop run

status:
	@set -e; \
	if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat "$(PID_FILE)" 2>/dev/null || true); \
		if [ -n "$$PID" ] && kill -0 $$PID 2>/dev/null; then \
			echo "Running (pid=$$PID)"; \
		else \
			echo "Not running (stale pid file: $$PID)"; \
		fi; \
	else \
		echo "Not running"; \
	fi
