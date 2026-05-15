# uv Documentation

> Offline documentation for [uv](https://docs.astral.sh/uv/) - An extremely fast Python package and project manager, written in Rust.

This folder contains curated documentation covering **project management**, **dependency handling**, and **GitHub Actions integration** with uv.

---

## Documentation Files

| File | Description |
|------|-------------|
| [uv-projects.md](./uv-projects.md) | Project creation, structure, configuration, and workspaces |
| [uv-dependencies.md](./uv-dependencies.md) | Dependency management, locking, syncing, and resolution |
| [uv-github-actions.md](./uv-github-actions.md) | CI/CD integration with GitHub Actions |

---

## How to Use This Documentation

### Finding Information

**For project setup and structure:**
- Creating new projects → `uv-projects.md` > Creating Projects
- Understanding `pyproject.toml` → `uv-projects.md` > Project Structure and Files
- Configuring entry points/scripts → `uv-projects.md` > Configuring Projects
- Multi-package repositories → `uv-projects.md` > Using Workspaces

**For dependency management:**
- Adding/removing packages → `uv-dependencies.md` > Managing Dependencies
- Git/URL/path dependencies → `uv-dependencies.md` > Dependency Sources
- Development dependencies → `uv-dependencies.md` > Optional and Development Dependencies
- Environment sync issues → `uv-dependencies.md` > Locking and Syncing
- Version conflicts → `uv-dependencies.md` > Resolution

**For CI/CD:**
- Basic GitHub Actions setup → `uv-github-actions.md` > Installation
- Matrix testing → `uv-github-actions.md` > Multiple Python Versions
- Caching strategies → `uv-github-actions.md` > Caching
- Publishing to PyPI → `uv-github-actions.md` > Publishing to PyPI

**For WSL setup and local workflows:**
- Install/PATH troubleshooting, venv usage, dependency install options, scripts, and tests → this file > Using `uv` in WSL

### Quick Command Reference

```bash
# Project Management
uv init [name]              # Create new project
uv init --lib [name]        # Create library project
uv run [script.py]          # Run in project environment
uv build                    # Build distributions

# Dependencies
uv add <package>            # Add dependency
uv add --dev <package>      # Add dev dependency
uv remove <package>         # Remove dependency
uv sync                     # Sync environment with lockfile
uv lock                     # Update lockfile
uv lock --upgrade           # Upgrade all packages

# Environment
uv python install           # Install Python version from .python-version
uv venv                     # Create virtual environment
```

---

## Using `uv` in WSL: install, PATH setup, venvs, dependencies, scripts, and tests

### 1) Install `uv` (WSL)

Recommended installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:

```bash
uv --version
command -v uv
```

If `uv --version` fails, `uv` is often installed but not on your `PATH`.

### 2) Fix `PATH` in WSL (important)

Common install path:

- `~/.local/bin/uv`

For the current terminal session:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
command -v uv
```

Make it permanent for bash in WSL:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Troubleshooting note:

- `source "$HOME/.local/bin/env"` may fail because `~/.local/bin/env` does not exist on all systems. Use PATH setup as shown above.

Optional location checks:

```bash
ls -la ~/.local/bin/uv 2>/dev/null
ls -la ~/.cargo/bin/uv 2>/dev/null
```

If `uv` is in `~/.cargo/bin` instead:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### 3) Create and activate a Python virtual environment with `uv`

Create a project folder and venv:

```bash
mkdir -p ~/proj/myapp && cd ~/proj/myapp
uv venv .venv
```

Activate it (bash/zsh):

```bash
source .venv/bin/activate
python -V
```

Without activation:

```bash
.venv/bin/python -V
```

### 4) Install dependencies

Use one approach based on project layout.

For this repository, prefer:

```bash
uv sync
```

If your project uses `pyproject.toml` and editable installs:

```bash
source .venv/bin/activate
uv pip install -e .
```

If your project defines a test extra:

```bash
uv pip install -e ".[test]"
```

If your project uses `requirements.txt`:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

Ad-hoc installs:

```bash
source .venv/bin/activate
uv pip install requests
```

Useful package commands:

```bash
uv pip list
uv pip freeze > requirements.txt
```

### 5) Run Python scripts

With activated venv:

```bash
source .venv/bin/activate
python path/to/script.py
```

Without activation:

```bash
.venv/bin/python path/to/script.py
```

Run a module:

```bash
python -m your_package.some_module
```

### 6) Run unit tests

Option 1: `pytest` (most common)

Install if needed:

```bash
source .venv/bin/activate
uv pip install pytest
```

Run:

```bash
pytest
```

Common variants:

```bash
pytest -q
pytest -k "name_contains"
pytest tests/test_something.py::test_one_thing
```

Option 2: built-in `unittest`

```bash
source .venv/bin/activate
python -m unittest
```

Discovery example:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 7) Quick sanity check: confirm you’re using the venv Python

```bash
python -c "import sys; print(sys.executable)"
```

If the path is under `.venv/`, your environment is active.

---

## Topics Not Fully Covered in This Documentation

The following topics are not covered in depth here. Use the official documentation for full detail:

### Tools & Scripts
- **Using tools (uvx)**: https://docs.astral.sh/uv/guides/tools/
- **Publishing packages**: https://docs.astral.sh/uv/guides/publish/

### Advanced Concepts
- **Python version management**: https://docs.astral.sh/uv/concepts/python-versions/
- **Package indexes**: https://docs.astral.sh/uv/concepts/indexes/
- **Authentication**: https://docs.astral.sh/uv/concepts/authentication/
- **Caching**: https://docs.astral.sh/uv/concepts/cache/
- **Advanced pip interface behavior**: https://docs.astral.sh/uv/concepts/pip/

### Other Integrations
- **Docker**: https://docs.astral.sh/uv/guides/integration/docker/
- **Jupyter**: https://docs.astral.sh/uv/guides/integration/jupyter/
- **Pre-commit**: https://docs.astral.sh/uv/guides/integration/pre-commit/
- **GitLab CI/CD**: https://docs.astral.sh/uv/guides/integration/gitlab/
- **PyTorch**: https://docs.astral.sh/uv/guides/integration/pytorch/
- **FastAPI**: https://docs.astral.sh/uv/guides/integration/fastapi/

### Reference
- **CLI reference**: https://docs.astral.sh/uv/reference/cli/
- **Settings reference**: https://docs.astral.sh/uv/reference/settings/
- **Environment variables**: https://docs.astral.sh/uv/reference/environment/
- **Troubleshooting**: https://docs.astral.sh/uv/reference/troubleshooting/

---

## Official Resources

- **Official Documentation**: https://docs.astral.sh/uv/
- **GitHub Repository**: https://github.com/astral-sh/uv
- **setup-uv Action**: https://github.com/astral-sh/setup-uv
- **Changelog**: https://github.com/astral-sh/uv/blob/main/CHANGELOG.md

---

## Documentation Version

- **Source**: docs.astral.sh/uv
- **Downloaded**: December 2025
- **uv Version Coverage**: 0.9.x

> **Note**: uv is actively developed. For the latest features and changes, always check the official documentation.
