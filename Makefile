.PHONY: dev stop

dev:
	@echo "Starting the AI Diagnostic Agent server..."
	uv run uvicorn src.server:app --reload --port 8000

stop:
	@echo "Stopping Uvicorn server processes..."
	pkill -f "uvicorn src.server:app" || echo "Server is not currently running."
