# uv GitHub Actions Integration

> Source: [docs.astral.sh/uv/guides/integration/github/](https://docs.astral.sh/uv/guides/integration/github/)

This document covers integrating uv with GitHub Actions for CI/CD workflows.

---

## Table of Contents

1. [Installation](#installation)
2. [Setting up Python](#setting-up-python)
3. [Multiple Python Versions](#multiple-python-versions)
4. [Syncing and Running](#syncing-and-running)
5. [Caching](#caching)
6. [Using uv pip](#using-uv-pip)
7. [Private Repos](#private-repos)
8. [Publishing to PyPI](#publishing-to-pypi)

---

## Installation

The official [`astral-sh/setup-uv`](https://github.com/astral-sh/setup-uv) action is recommended for GitHub Actions workflows. It installs uv, adds it to PATH, optionally persists the cache, and supports all uv-supported platforms.

### Latest Version

```yaml
name: Example

jobs:
  uv-example:
    name: python
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7
```

### Pinned Version

Best practice involves pinning to a specific uv version:

```yaml
name: Example

jobs:
  uv-example:
    name: python
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          # Install a specific version of uv.
          version: "0.9.14"
```

---

## Setting up Python

### Using uv to Install Python

Install Python using the `uv python install` command, which respects the project's pinned version:

```yaml
name: Example

jobs:
  uv-example:
    name: python
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Set up Python
        run: uv python install
```

### Using actions/setup-python

Alternatively, use the official GitHub `setup-python` action, which may be faster due to cached Python versions.

**With `.python-version` file:**

```yaml
name: Example

jobs:
  uv-example:
    name: python
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v5

      - name: "Set up Python"
        uses: actions/setup-python@v6
        with:
          python-version-file: ".python-version"

      - name: Install uv
        uses: astral-sh/setup-uv@v7
```

**With `pyproject.toml` file:**

```yaml
name: Example

jobs:
  uv-example:
    name: python
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v5

      - name: "Set up Python"
        uses: actions/setup-python@v6
        with:
          python-version-file: "pyproject.toml"

      - name: Install uv
        uses: astral-sh/setup-uv@v7
```

---

## Multiple Python Versions

### Using setup-uv with Matrix Strategy

Set the Python version using `astral-sh/setup-uv` to override specifications in configuration files:

```yaml
jobs:
  build:
    name: continuous-integration
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version:
          - "3.10"
          - "3.11"
          - "3.12"

    steps:
      - uses: actions/checkout@v5

      - name: Install uv and set the Python version
        uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
```

### Using UV_PYTHON Environment Variable

Without the `setup-uv` action, set the `UV_PYTHON` environment variable:

```yaml
jobs:
  build:
    name: continuous-integration
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version:
          - "3.10"
          - "3.11"
          - "3.12"
    env:
      UV_PYTHON: ${{ matrix.python-version }}
    steps:
      - uses: actions/checkout@v5
```

---

## Syncing and Running

Install the project with `uv sync` and execute commands in the environment:

```yaml
name: Example

jobs:
  uv-example:
    name: python
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Install the project
        run: uv sync --locked --all-extras --dev

      - name: Run tests
        # For example, using `pytest`
        run: uv run pytest tests
```

**Tip:** The `UV_PROJECT_ENVIRONMENT` setting allows installation to the system Python environment instead of creating a virtual environment.

---

## Caching

### Built-in Cache Support

The `astral-sh/setup-uv` action includes built-in caching support:

```yaml
- name: Enable caching
  uses: astral-sh/setup-uv@v7
  with:
    enable-cache: true
```

### Manual Cache Management

Alternatively, manage caching manually using `actions/cache`:

```yaml
jobs:
  install_job:
    env:
      # Configure a constant location for the uv cache
      UV_CACHE_DIR: /tmp/.uv-cache

    steps:
      # ... setup up Python and uv ...

      - name: Restore uv cache
        uses: actions/cache@v4
        with:
          path: /tmp/.uv-cache
          key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
          restore-keys: |
            uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
            uv-${{ runner.os }}

      # ... install packages, run tests, etc ...

      - name: Minimize uv cache
        run: uv cache prune --ci
```

The `uv cache prune --ci` command reduces cache size and is optimized for CI environments.

**Note:** When using `uv pip`, use `requirements.txt` instead of `uv.lock` in the cache key.

### Self-Hosted Runners

For non-ephemeral, self-hosted runners, place the cache inside the GitHub Workspace:

```yaml
install_job:
  env:
    # Configure a relative location for the uv cache
    UV_CACHE_DIR: ${{ github.workspace }}/.cache/uv
```

Clean the cache after each job using a post job hook with a cleanup script:

```bash
#!/usr/bin/env sh
uv cache clean
```

---

## Using uv pip

When using the `uv pip` interface instead of the project interface, uv requires a virtual environment by default. Use the `--system` flag or set `UV_SYSTEM_PYTHON` to install into the system environment.

### Workflow-Level Setting

```yaml
env:
  UV_SYSTEM_PYTHON: 1

jobs: ...
```

### Job-Level Setting

```yaml
jobs:
  install_job:
    env:
      UV_SYSTEM_PYTHON: 1
    ...
```

### Step-Level Setting

```yaml
steps:
  - name: Install requirements
    run: uv pip install -r requirements.txt
    env:
      UV_SYSTEM_PYTHON: 1
```

To opt-out, use the `--no-system` flag in any uv invocation.

---

## Private Repos

For projects with dependencies on private GitHub repositories, create a personal access token (PAT) with read access to those repositories and add it as a repository secret.

Configure a Git credential helper using the `gh` CLI:

```yaml
steps:
  - name: Register the personal access token
    run: echo "${{ secrets.MY_PAT }}" | gh auth login --with-token
  - name: Configure the Git credential helper
    run: gh auth setup-git
```

---

## Publishing to PyPI

uv can build and publish packages to PyPI from GitHub Actions using trusted publishing (no credentials required).

### Release Workflow

```yaml
name: "Publish"

on:
  push:
    tags:
      # Publish on any tag starting with a `v`, e.g., v0.1.0
      - v*

jobs:
  run:
    runs-on: ubuntu-latest
    environment:
      name: pypi
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Checkout
        uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Install Python 3.13
        run: uv python install 3.13

      - name: Build
        run: uv build

      # Check that basic features work and we didn't miss to include crucial files
      - name: Smoke test (wheel)
        run: uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py

      - name: Smoke test (source distribution)
        run: uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py

      - name: Publish
        run: uv publish
```

### Configuration Steps

1. Create the "pypi" environment in repository settings under **Settings** → **Environments**
2. Add a trusted publisher to the PyPI project settings under **Publishing**
3. Ensure all fields match your GitHub configuration
4. Tag and push a release starting with `v`:

```bash
$ git tag -a v0.1.0 -m v0.1.0
$ git push --tags
```

---

## Complete Example Workflow

Here's a complete CI workflow combining multiple features:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          version: "0.9.14"
          enable-cache: true
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --locked --all-extras --dev

      - name: Run linting
        run: uv run ruff check .

      - name: Run type checking
        run: uv run mypy .

      - name: Run tests
        run: uv run pytest tests -v

  build:
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Build package
        run: uv build

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

---

## Quick Reference

| Task | Command/Setting |
|------|-----------------|
| Install uv | `uses: astral-sh/setup-uv@v7` |
| Pin uv version | `with: version: "0.9.14"` |
| Set Python version | `with: python-version: "3.12"` |
| Enable caching | `with: enable-cache: true` |
| Install project | `uv sync --locked --all-extras --dev` |
| Run tests | `uv run pytest tests` |
| Build package | `uv build` |
| Publish to PyPI | `uv publish` |
| Prune cache for CI | `uv cache prune --ci` |
| Use system Python | `env: UV_SYSTEM_PYTHON: 1` |
