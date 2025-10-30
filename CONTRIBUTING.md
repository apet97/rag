Contributing

- Install tooling:
  - python3 -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt -c constraints.txt
  - pre-commit install

- Workflow:
  - Create a feature branch
  - Run: pre-commit run --all-files
  - Run tests: OFFLINE_MODE=1 MOCK_LLM=true pytest -q
  - Open a PR; ensure CI is green

- Code style: black + isort, flake8 line length 100
- Security: run bandit and pip-audit locally before PR
