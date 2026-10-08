# Document Clustering FlaskRestFul API

Document categorization experiments with a Streamlit interface and a Flask SQLite cluster-management API.

## Repository guide

### Contents

- [LICENSE](LICENSE)
- [README.md](README.md)
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

### Maintenance fixes

- Preserve existing SQLite data during initialization.
- Initialize the database when imported by a Flask server.
- Validate assignment payloads and enforce foreign keys.

### Validation

Reviewed on 2026-10-08. Two regression tests passed with unittest. Python source syntax checks passed. See tests/ for the tested behavior.

```bash
python -m unittest discover -s tests -v
```

### Contributions

Describe the issue, reproduction steps, environment, and expected behavior when proposing a change. Keep generated environments, credentials, and unnecessary build artifacts out of new commits.

### License

See [LICENSE](LICENSE) for the repository’s licensing terms.
