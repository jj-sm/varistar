.PHONY: lint format-check test check release

# Derived from the origin remote so `gh` calls work regardless of the
# invoking shell's `gh repo set-default` state.
GH_REPO := $(shell gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)

# --- Local checks -----------------------------------------------------------
# `check` mirrors exactly what .github/workflows/publish.yml gates a release
# on: ruff check + pytest. `format-check` mirrors lint.yml's formatting
# check, which is advisory only there (continue-on-error: true), so it's
# kept separate and not part of the release-blocking `check` target.

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

test:
	uv run pytest

check: lint test

# --- Release ------------------------------------------------------------
# Usage: make release VERSION=0.1.12
#
# Requires CHANGELOG.md to already contain a "## v<VERSION>" entry,
# committed on main, before anything else runs. Then runs lint+tests
# locally, bumps the version in pyproject.toml and
# src/varistar/__init__.py, commits, pushes to main, tags, pushes the tag
# (which triggers .github/workflows/publish.yml to test + publish to PyPI),
# waits for that workflow to finish, and only then creates the GitHub
# release — so a release is never created for a version that failed to
# publish.

release:
ifndef VERSION
	$(error VERSION is not set. Usage: make release VERSION=0.1.12)
endif
	@echo "==> Checking working tree is clean and on main"
	@git diff --quiet && git diff --cached --quiet || (echo "Working tree has uncommitted changes. Commit or stash first." && exit 1)
	@[ "$$(git rev-parse --abbrev-ref HEAD)" = "main" ] || (echo "Not on main branch." && exit 1)
	@git fetch origin main --quiet
	@[ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" ] || (echo "Local main is not up to date with origin/main. Pull first." && exit 1)

	@echo "==> Checking CHANGELOG.md has an entry for v$(VERSION)"
	@grep -q "^## v$(VERSION)" CHANGELOG.md || (echo "CHANGELOG.md has no '## v$(VERSION)' entry. Add and commit one before releasing." && exit 1)

	@echo "==> Running lint + tests locally"
	$(MAKE) check

	@echo "==> Bumping version to $(VERSION)"
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	@sed -i.bak 's/^__version__ = ".*"/__version__ = "$(VERSION)"/' src/varistar/__init__.py && rm src/varistar/__init__.py.bak

	@echo "==> Committing"
	git add pyproject.toml src/varistar/__init__.py
	git commit -m "chore: release v$(VERSION)"

	@echo "==> Pushing to main"
	git push origin main

	@echo "==> Tagging v$(VERSION)"
	git tag v$(VERSION)
	git push origin v$(VERSION)

	@echo "==> Waiting for publish workflow on tag v$(VERSION)"
	@sleep 5
	@run_id=""; \
	for i in $$(seq 1 30); do \
		run_id=$$(gh run list --repo $(GH_REPO) --workflow=publish.yml --branch v$(VERSION) --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null); \
		[ -n "$$run_id" ] && break; \
		sleep 2; \
	done; \
	if [ -z "$$run_id" ]; then \
		echo "Could not find a publish.yml run for tag v$(VERSION). Check GitHub Actions manually."; \
		exit 1; \
	fi; \
	echo "Watching run $$run_id"; \
	gh run watch $$run_id --repo $(GH_REPO) --exit-status; \
	if [ $$? -ne 0 ]; then \
		echo "Publish workflow failed. Not creating a GitHub release."; \
		exit 1; \
	fi

	@echo "==> Creating GitHub release v$(VERSION)"
	gh release create v$(VERSION) --repo $(GH_REPO) --title "v$(VERSION)" --generate-notes

	@echo "==> Done. Released v$(VERSION) to PyPI and GitHub."
