# Contributing

Thanks for your interest in contributing! This project is a practical example of building an AI Travel Agent with LangGraph and multiple LLM tools. Contributions that improve reliability, developer experience, documentation, or functionality are welcome.

## Getting Started
- Ensure you have Python `3.11` installed.
- Install dependencies using Poetry:
  - `poetry install`
  - `poetry shell`
- Copy `.env.example` to `.env` and set the required keys.

## Development Workflow
- Create a feature branch from `main`.
- Run tests locally: `pytest`
- Lint your changes: `ruff check .`
- Keep changes focused and minimal; write clear commit messages.

## Pull Requests
- Fill in the PR template for context.
- Ensure CI passes (lint + tests).
- Reference related issues and add screenshots if UI output changes in Streamlit.

## Commit Message Tips
- Use imperative mood (e.g., "Add", "Fix", "Update").
- Prefer small, self-contained commits.

## Reporting Issues
- Use the Bug Report or Feature Request templates.
- Provide reproduction steps and environment details.

## Code Style
- Follow Pythonic conventions; keep functions small and clear.
- Avoid adding new dependencies without discussion; prefer built-ins or existing libs.

## Security and Secrets
- Never commit secrets. Use `.env` locally and GitHub Secrets in CI.

## License
- By contributing, you agree your contributions are licensed under the repository’s MIT license.