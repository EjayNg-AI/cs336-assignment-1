# uv Projects Documentation

> Source: [docs.astral.sh/uv](https://docs.astral.sh/uv/)

This document covers project creation, structure, configuration, and workspaces in uv.

---

## Table of Contents

1. [Working on Projects](#working-on-projects)
2. [Project Structure and Files](#project-structure-and-files)
3. [Creating Projects](#creating-projects)
4. [Configuring Projects](#configuring-projects)
5. [Using Workspaces](#using-workspaces)
6. [Building Distributions](#building-distributions)

---

## Working on Projects

### Creating a New Project

Initialize a new Python project using the `uv init` command:

```bash
$ uv init hello-world
$ cd hello-world
```

Or initialize in your current directory:

```bash
$ mkdir hello-world
$ cd hello-world
$ uv init
```

uv generates these files:
- `.gitignore`
- `.python-version`
- `README.md`
- `main.py`
- `pyproject.toml`

Run your project with `uv run`:

```bash
$ uv run main.py
Hello from hello-world!
```

### Project Structure

A complete project includes:

```
.
├── .venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

#### pyproject.toml

Contains project metadata and dependencies:

```toml
[project]
name = "hello-world"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
dependencies = []
```

Use this file to manage dependencies and configure uv settings via a `[tool.uv]` section.

#### .python-version

Specifies the project's default Python version for virtual environment creation.

#### .venv

Isolated Python environment where dependencies are installed.

#### uv.lock

Cross-platform lockfile containing exact dependency versions. It's human-readable but should not be edited manually. Check this into version control for reproducible installations.

### Viewing Your Version

Display package version information:

```bash
$ uv version
hello-world 0.7.0

$ uv version --short
0.7.0

$ uv version --output-format json
{
    "package_name": "hello-world",
    "version": "0.7.0",
    "commit_info": null
}
```

### Building Distributions

Generate source distributions and wheels:

```bash
$ uv build
$ ls dist/
hello-world-0.1.0-py3-none-any.whl
hello-world-0.1.0.tar.gz
```

---

## Project Structure and Files

### The `pyproject.toml`

Python project metadata is defined in a `pyproject.toml` file. uv requires this file to identify the root directory of a project.

The `uv init` command creates new projects. A minimal project needs:

```toml
[project]
name = "example"
version = "0.1.0"
```

Additional configuration covers Python version requirements, dependencies, build systems, and entry points.

### The Project Environment

When developing with uv, the tool automatically creates a virtual environment. Most uv commands generate temporary environments, while uv also maintains a persistent environment containing the project and dependencies in a `.venv` directory adjacent to `pyproject.toml`.

It is stored inside the project to make it easy for editors to find — they need the environment to give code completions and type hints.

The `.venv` directory is excluded from version control automatically. Execute commands using `uv run` or activate the environment normally. The `uv run` command creates or updates the project environment as needed, or you can explicitly create it with `uv sync`.

**Best practices:** Avoid manual environment modifications. Use `uv add` for project dependencies, `uvx` for one-off tools, or `uv run --with` for temporary requirements.

Disable automatic management with `[tool.uv] managed = false` in `pyproject.toml` if desired.

### The Lockfile

uv creates a `uv.lock` file next to the `pyproject.toml`.

The `uv.lock` represents a *universal* or *cross-platform* lockfile that captures the packages that would be installed across all possible Python markers.

Unlike `pyproject.toml`, which specifies broad requirements, the lockfile contains exact resolved versions. This file should be committed to version control for reproducible installations across environments and consistent deployments.

The lockfile is automatically created and updated during uv invocations that use the project environment, i.e., `uv sync` and `uv run`.

The format is human-readable TOML but managed by uv—manual edits are discouraged.

#### Relationship to `pylock.toml`

In PEP 751, Python standardized a new resolution file format, `pylock.toml`. This format replaces `requirements.txt` for resolution outputs and is tool-agnostic. However, uv continues using `uv.lock` for project workflows since some functionality cannot be expressed in `pylock.toml`.

uv supports exporting and using `pylock.toml`:

- Export: `uv export -o pylock.toml`
- Generate: `uv pip compile requirements.in -o pylock.toml`
- Install: `uv pip sync pylock.toml` or `uv pip install -r pylock.toml`

---

## Creating Projects

### Overview

The `uv init` command creates new Python projects. uv offers two primary templates: applications (default) and libraries, controlled via flags like `--lib`.

### Target Directory

Projects can be initialized in the current directory or a specified location using `uv init foo`. The `--directory` option allows changing the working directory for project creation. An error occurs if a `pyproject.toml` already exists in the target location.

### Applications

Application projects suit web servers, scripts, and CLIs. They're the default project type but can be explicitly specified with `--app`.

**Basic command:**
```bash
$ uv init example-app
```

**Generated structure:**
```
example-app
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```

The `pyproject.toml` contains basic metadata without a build system:

```toml
[project]
name = "example-app"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []
```

The `main.py` provides standard boilerplate with a main function. Files execute via `uv run main.py`.

### Packaged Applications

Packaged applications suit command-line tools for PyPI distribution or projects requiring dedicated test directories.

**Command:**
```bash
$ uv init --package example-pkg
```

**Structure:**
```
example-pkg
├── .python-version
├── README.md
├── pyproject.toml
└── src
    └── example_pkg
        └── __init__.py
```

A build system is included in `pyproject.toml`:

```toml
[build-system]
requires = ["uv_build>=0.9.14,<0.10.0"]
build-backend = "uv_build"
```

Entry points for CLI commands are defined:

```toml
[project.scripts]
example-pkg = "example_pkg:main"
```

Execute with `uv run example-pkg`.

### Libraries

Libraries distribute functions and objects for other projects. The `--lib` flag creates library projects, which automatically implies `--package`.

**Command:**
```bash
$ uv init --lib example-lib
```

**Structure:**
```
example-lib
├── .python-version
├── README.md
├── pyproject.toml
└── src
    └── example_lib
        ├── py.typed
        └── __init__.py
```

A `py.typed` marker indicates type availability to consumers. The src layout isolates the library from project root invocations.

The `__init__.py` defines a simple function:
```python
def hello() -> str:
    return "Hello from example-lib!"
```

Access via `uv run python -c "import example_lib; print(example_lib.hello())"`.

### Projects with Extension Modules

Non-pure-Python projects use extension modules written in C, C++, FORTRAN, or Rust for performance. uv supports build systems:

- **maturin** for Rust
- **scikit-build-core** for C, C++, FORTRAN, Cython

**Command:**
```bash
$ uv init --build-backend maturin example-ext
```

**Structure:**
```
example-ext
├── .python-version
├── Cargo.toml
├── README.md
├── pyproject.toml
└── src
    ├── lib.rs
    └── example_ext
        ├── __init__.py
        └── _core.pyi
```

The Rust library (`lib.rs`) provides compiled functionality, while Python modules wrap it.

**Important:** Configure `tool.uv.cache-keys` to include source file types. Use `--reinstall` to force rebuilds.

### Minimal Projects

The `--bare` option creates only `pyproject.toml` without version pins, READMEs, or source files:

```bash
$ uv init example --bare
```

**Result:**
```
example-bare
└── pyproject.toml
```

The minimal `pyproject.toml`:
```toml
[project]
name = "example"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
```

The `--bare` flag combines with `--lib` or `--build-backend`. Additional features enable opt-in via flags like `--description`, `--author-from git`, `--vcs git`, and `--python-pin`.

---

## Configuring Projects

### Python Version Requirements

Projects can declare supported Python versions using the `project.requires-python` field in `pyproject.toml`. This specification determines which Python syntax is permitted and affects dependency version selection.

```toml
[project]
name = "example"
version = "0.1.0"
requires-python = ">=3.12"
```

### Entry Points

Entry points allow installed packages to advertise interfaces. They require a build system to be defined.

#### Command-Line Interfaces

Define CLIs in the `[project.scripts]` table:

```toml
[project.scripts]
hello = "example:hello"
```

Execute with: `uv run hello`

#### Graphical User Interfaces

GUIs are defined in `[project.gui-scripts]`. On Windows, they're wrapped by GUI executables; on other platforms, they behave identically to CLIs.

```toml
[project.gui-scripts]
hello = "example:app"
```

#### Plugin Entry Points

Declare plugins in `[project.entry-points]`:

```toml
[project.entry-points.'example.plugins']
a = "example_plugin_a"
```

Load plugins using:

```python
from importlib.metadata import entry_points

for plugin in entry_points(group='example.plugins'):
    plugin.load()
```

### Build Systems

Build systems determine packaging and installation. Projects declare these in the `[build-system]` table. If undefined, uv won't build the project itself, only install dependencies.

Use `--build-backend` or `--package` flags with `uv init` to create packaged projects.

Build systems enable:
- File inclusion/exclusion in distributions
- Editable installation behavior
- Dynamic metadata
- Native code compilation
- Shared library vendoring

### Project Packaging

Package your project if you plan to:
- Add command-line tools
- Distribute publicly
- Use src/test layouts
- Write libraries

Skip packaging for scripts, simple applications, or flat layouts.

Override build system behavior with `tool.uv.package`:
- Set to `true` to force packaging even without a build system declaration
- Set to `false` to prevent packaging despite build system presence

### Project Environment Path

Use `UV_PROJECT_ENVIRONMENT` to configure the virtual environment path (defaults to `.venv`).

For system environments, set it to the Python installation prefix:

```bash
python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))"
```

**Note:** Writing to system environments is discouraged; `uv sync` removes extraneous packages and may break the system.

### Build Isolation

By default, uv builds packages in isolated environments alongside declared build dependencies per PEP 517.

#### Augmenting Build Dependencies

For packages requiring additional build dependencies, use `extra-build-dependencies`:

```toml
[tool.uv.extra-build-dependencies]
cchardet = ["cython"]
```

Enable version matching with `match-runtime = true`:

```toml
[tool.uv.extra-build-dependencies]
deepspeed = [{ requirement = "torch", match-runtime = true }]
```

#### Dynamic Metadata

When packages lack static metadata, specify exact versions and augment dependencies:

```toml
[project]
dependencies = ["axolotl[deepspeed, flash-attn]", "torch==2.6.0"]

[tool.uv.extra-build-dependencies]
axolotl = ["torch==2.6.0"]
deepspeed = ["torch==2.6.0"]
flash-attn = ["torch==2.6.0"]
```

#### Disabling Build Isolation

Mark packages to skip isolated builds via `no-build-isolation-package`:

```toml
[tool.uv]
no-build-isolation-package = ["cchardet"]
```

Build dependencies must be installed beforehand. Isolate them to optional groups:

```toml
[project]
dependencies = ["cchardet"]

[project.optional-dependencies]
build = ["setuptools", "cython"]

[tool.uv]
no-build-isolation-package = ["cchardet"]
```

Install with: `uv sync --extra build`, then `uv sync` to remove build dependencies.

### Editable Mode

Projects install in editable mode by default, reflecting source code changes immediately. Use `--no-editable` for non-editable installation in deployment contexts.

### Conflicting Dependencies

Declare incompatible dependency groups explicitly:

```toml
[tool.uv]
conflicts = [
    [
      { extra = "extra1" },
      { extra = "extra2" },
    ],
]
```

### Limited Resolution Environments

Constrain supported platforms using PEP 508 environment markers:

```toml
[tool.uv]
environments = [
    "sys_platform == 'darwin'",
    "sys_platform == 'linux'",
]
```

### Required Environments

Mark mandatory platform/version support:

```toml
[tool.uv]
required-environments = [
    "sys_platform == 'darwin' and platform_machine == 'x86_64'",
]
```

This applies primarily to packages without source distributions.

---

## Using Workspaces

### Overview

A workspace, inspired by Rust's Cargo, is a collection of one or more packages, called *workspace members*, that are managed together. Workspaces allow developers to organize large codebases by dividing them into multiple packages sharing common dependencies, all within a single Git repository.

Key characteristic: While each package maintains its own `pyproject.toml`, the workspace operates with a single shared lockfile ensuring consistent dependency versions across all members.

### Getting Started

To create a workspace, add a `tool.uv.workspace` table to a `pyproject.toml`:

```toml
[project]
name = "albatross"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["bird-feeder", "tqdm>=4,<5"]

[tool.uv.sources]
bird-feeder = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]
exclude = ["packages/seeds"]
```

**Requirements:**
- Every directory matched by `members` globs must contain a `pyproject.toml`
- Workspace members can be applications or libraries
- Every workspace needs a root (which is also a member)
- `uv run` and `uv sync` operate on the workspace root by default

### Workspace Sources

Dependencies on workspace members use `tool.uv.sources` with the `workspace = true` key:

```toml
[project]
name = "albatross"
dependencies = ["bird-feeder", "tqdm>=4,<5"]

[tool.uv.sources]
bird-feeder = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]
```

**Important notes:**
- Dependencies between workspace members are editable
- `tool.uv.sources` definitions in the workspace root apply to all members unless overridden
- Member-specific overrides take precedence, even with platform markers

### Workspace Layouts

Common structure with root project and accompanying libraries:

```
albatross
├── packages
│   ├── bird-feeder
│   │   ├── pyproject.toml
│   │   └── src
│   │       └── bird_feeder
│   │           ├── __init__.py
│   │           └── foo.py
│   └── seeds
│       ├── pyproject.toml
│       └── src
│           └── seeds
│               ├── __init__.py
│               └── bar.py
├── pyproject.toml
├── README.md
├── uv.lock
└── src
    └── albatross
        └── main.py
```

### When (Not) to Use Workspaces

**Ideal use cases:**
- Multiple interconnected packages within one repository
- Libraries with performance-critical extensions (Rust, C++)
- Libraries with plugin systems
- Testing core functionality independently from CLI

**Not suitable for:**
- Members with conflicting dependency requirements
- Cases needing separate virtual environments per member
- Python version conflicts across members

**Alternatives:** Use path dependencies via `tool.uv.sources` for independent projects:

```toml
[tool.uv.sources]
bird-feeder = { path = "packages/bird-feeder" }
```

**Limitations:** Workspaces enforce a single `requires-python` value as the intersection of all members' requirements. If testing specific members on unsupported Python versions is necessary, use `uv pip` with separate virtual environments.

---

## Building Distributions

uv provides `uv build` to create distributable formats for Python projects.

### Using uv build

The basic command builds both source distributions (sdists) and binary distributions (wheels):

```bash
$ uv build
$ ls dist/
example-0.1.0-py3-none-any.whl
example-0.1.0.tar.gz
```

Build options include:
- `uv build path/to/project` — build a specific project directory
- `uv build --sdist` — source distribution only
- `uv build --wheel` — binary distribution only
- `uv build --sdist --wheel` — both from source

### Build Constraints

Use `--build-constraint` to control build requirement versions. Combined with `--require-hashes`, this ensures reproducible builds:

```bash
$ uv build --build-constraint constraints.txt --require-hashes
```

Example constraint file format:
```
setuptools==68.2.2 --hash=sha256:[hash_value]
```

### Preventing PyPI Publication

Mark internal packages as private to prevent accidental PyPI uploads:

```toml
[project]
classifiers = ["Private :: Do Not Upload"]
```

Additionally, use per-project PyPI API tokens to add security protection against unauthorized publishing attempts.

---

## Quick Reference

| Task | Command |
|------|---------|
| Create new project | `uv init [name]` |
| Create library | `uv init --lib [name]` |
| Create packaged app | `uv init --package [name]` |
| Run project | `uv run [script.py]` |
| Build distributions | `uv build` |
| Build sdist only | `uv build --sdist` |
| Build wheel only | `uv build --wheel` |
| Check version | `uv version` |
| Sync environment | `uv sync` |
