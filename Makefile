.PHONY: test lint format fixtures
test:
	uv run pytest -q
lint:
	uv run ruff check src tests && uv run ruff format --check src tests
format:
	uv run ruff format src tests && uv run ruff check --fix src tests
# Regenerate reference fixtures. Needs R with mgcv; CI does this inside the pinned oracle image.
fixtures:
	Rscript oracle/make_fixtures.R tests/fixtures
