.PHONY: install lint

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files
