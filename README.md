# Document Clustering FlaskRestFul API

Document categorization experiments with a Streamlit interface and a Flask SQLite cluster-management API.

## Setup and repository reference

### Project structure

- [LICENSE](LICENSE)
- [Task_2](Task_2)
- [requirements.txt](requirements.txt)
- [streamlit_app.py](streamlit_app.py)
- [tests](tests)
- [text_classifier.db](text_classifier.db)

### Getting started

```bash
git clone https://github.com/Raimal-Raja/Document_Clustering_FlaskRestFul_API.git
cd Document_Clustering_FlaskRestFul_API
```

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r "requirements.txt"
```

Application entry point:

```bash
python Task_2/app.py
```

### Configuration and limitations

The Flask cluster-management API and Streamlit categorization interface are separate examples. CLUSTERING_DB_PATH overrides the Flask SQLite database location. Database/API regression tests do not validate the Streamlit classifier.

### Maintenance fixes

- Preserve existing SQLite data during initialization.
- Initialize the database when imported by a Flask server.
- Validate assignment payloads and enforce foreign keys.

### Validation

Audit: 2026-10-08. Repository structure, setup instructions and description were reviewed. 3 existing Python files passed syntax checks; changed files and new regression tests were checked separately. 2 regression tests passed. Syntax checks do not establish full runtime correctness. External APIs, live scraping, GUI interaction, notebook training and production deployment were not comprehensively exercised.

```bash
python -m unittest discover -s tests -v
```

### Repository description

The short GitHub description is provided in [REPOSITORY_DESCRIPTION.md](REPOSITORY_DESCRIPTION.md).

### Contributions

Describe the issue, reproduction steps, environment, and expected behavior when proposing a change. Keep generated environments, credentials, and unnecessary build artifacts out of new commits.

### License

See [LICENSE](LICENSE) for the repository’s licensing terms.
