# Contributing to Text Complaint API

Thank you for taking the time to contribute! The following guidelines help keep the project consistent and high quality.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Running Tests](#running-tests)

---

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/text-complaint-api.git
   cd text-complaint-api
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8
   ```

3. Copy the environment template and fill in your values:
   ```bash
   cp .env.example .env   # Edit .env with your HF_TOKEN and thresholds
   ```

---

## Development Workflow

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes with clear, focused commits.

3. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

4. Push your branch and open a Pull Request against `master`.

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
<type>(<scope>): <short summary>
```

Common types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `ci`.

Examples:
- `feat(pipeline): add confidence score logging`
- `fix(rule-engine): handle missing intent label`
- `docs(readme): update setup instructions`

---

## Pull Requests

- Fill in the PR template completely.
- Reference the related issue (e.g., `Closes #42`).
- Keep PRs focused — one logical change per PR.
- All CI checks must pass before merging.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

To check coverage:
```bash
pytest tests/ --cov=. --cov-report=term-missing
```
