# uv Dependencies Documentation

> Source: [docs.astral.sh/uv](https://docs.astral.sh/uv/)

This document covers dependency management, locking, syncing, resolution, and exporting in uv.

---

## Table of Contents

1. [Managing Dependencies](#managing-dependencies)
2. [Dependency Sources](#dependency-sources)
3. [Optional and Development Dependencies](#optional-and-development-dependencies)
4. [Locking and Syncing](#locking-and-syncing)
5. [Running Commands](#running-commands)
6. [Resolution](#resolution)
7. [Exporting Lockfiles](#exporting-lockfiles)

---

## Managing Dependencies

### Dependency Fields

Projects define dependencies across several fields:

- **`project.dependencies`**: Published dependencies
- **`project.optional-dependencies`**: Published optional dependencies ("extras")
- **`dependency-groups`**: Local development dependencies
- **`tool.uv.sources`**: Alternative sources for dependencies during development

Note: The `project.dependencies` and `project.optional-dependencies` fields work even for unpublished projects. The `dependency-groups` feature is recently standardized and may lack universal tool support.

uv can modify dependencies via `uv add` and `uv remove` commands, or by directly editing `pyproject.toml`.

### Adding Dependencies

To add a dependency:

```bash
$ uv add httpx
```

This creates an entry in `project.dependencies`:

```toml
[project]
name = "example"
version = "0.1.0"
dependencies = ["httpx>=0.27.2"]
```

Use `--dev`, `--group`, or `--optional` flags to add dependencies to alternative fields. Version constraints can be customized with `--bounds` or specified directly:

```bash
$ uv add "httpx>=0.20"
```

When adding from non-registry sources, entries appear in `tool.uv.sources`. For example, adding from GitHub:

```bash
$ uv add "httpx @ git+https://github.com/encode/httpx"
```

Results in:

```toml
[project]
name = "example"
version = "0.1.0"
dependencies = ["httpx"]

[tool.uv.sources]
httpx = { git = "https://github.com/encode/httpx" }
```

### Importing Dependencies from Requirements Files

Import dependencies from `requirements.txt`:

```bash
uv add -r requirements.txt
```

### Removing Dependencies

To remove a dependency:

```bash
$ uv remove httpx
```

Use `--dev`, `--group`, or `--optional` flags to remove from specific tables. Associated sources are also removed if no other references exist.

### Changing Dependencies

Update an existing dependency constraint:

```bash
$ uv add "httpx>0.1.0"
```

The locked version only changes if necessary to satisfy new constraints. To force updating to the latest compatible version:

```bash
$ uv add "httpx>0.1.0" --upgrade-package httpx
```

Change source by specifying a different location:

```bash
$ uv add "httpx @ ../httpx"
```

### Platform-Specific Dependencies

Use environment markers to restrict dependencies to specific platforms or Python versions:

```bash
$ uv add "jax; sys_platform == 'linux'"
$ uv add "numpy; python_version >= '3.11'"
```

Results in:

```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["jax; sys_platform == 'linux'"]
```

### Project Dependencies

The `project.dependencies` table represents dependencies for publication to PyPI or wheel building. Uses standard dependency specifier syntax per PEP 621:

```toml
[project]
name = "albatross"
version = "0.1.0"
dependencies = [
  "tqdm >=4.66.2,<5",
  "torch ==2.2.2",
  "transformers[torch] >=4.39.3,<5",
  "importlib_metadata >=7.1.0,<8; python_version < '3.10'",
  "mollymawk ==0.1.0"
]
```

---

## Dependency Sources

The `tool.uv.sources` table extends standard dependency tables with alternative sources used during development. Example:

```toml
[project]
name = "example"
version = "0.1.0"
dependencies = ["foo"]

[tool.uv.sources]
foo = { path = "./packages/foo" }
```

**Important**: Sources are respected only by uv. Other tools use only standard project definitions.

### Index

Add packages from specific indexes:

```bash
$ uv add torch --index pytorch=https://download.pytorch.org/whl/cpu
```

Results in:

```toml
[project]
dependencies = ["torch"]

[tool.uv.sources]
torch = { index = "pytorch" }

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
```

An `explicit` flag restricts the index to packages that explicitly specify it:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

### Git

Add Git dependencies using `git+` prefix:

```bash
$ uv add git+https://github.com/encode/httpx
$ uv add git+ssh://[email protected]/encode/httpx
```

Results in:

```toml
[project]
dependencies = ["httpx"]

[tool.uv.sources]
httpx = { git = "https://github.com/encode/httpx" }
```

Specify Git references with tags, branches, or commits:

```bash
$ uv add git+https://github.com/encode/httpx --tag 0.27.0
$ uv add git+https://github.com/encode/httpx --branch main
$ uv add git+https://github.com/encode/httpx --rev 326b9431c761e1ef1e00b9f760d1f654c8db48c6
```

Specify subdirectories:

```bash
$ uv add git+https://github.com/langchain-ai/langchain#subdirectory=libs/langchain
```

### URL

Add remote wheels or source distributions:

```bash
$ uv add "https://files.pythonhosted.org/packages/.../httpx-0.27.0.tar.gz"
```

Results in:

```toml
[project]
dependencies = ["httpx"]

[tool.uv.sources]
httpx = { url = "https://files.pythonhosted.org/packages/.../httpx-0.27.0.tar.gz" }
```

### Path

Add local wheels, source distributions, or project directories:

```bash
$ uv add /example/foo-0.1.0-py3-none-any.whl
$ uv add ./foo-0.1.0-py3-none-any.whl
$ uv add ~/projects/bar/
```

Results in:

```toml
[project]
dependencies = ["foo"]

[tool.uv.sources]
foo = { path = "/example/foo-0.1.0-py3-none-any.whl" }
```

For directory dependencies, uv builds and installs them by default. Request editable installation:

```bash
$ uv add --editable ../projects/bar/
```

Results in:

```toml
[project]
dependencies = ["bar"]

[tool.uv.sources]
bar = { path = "../projects/bar", editable = true }
```

### Workspace Member

Declare workspace member dependencies with `{ workspace = true }`:

```toml
[project]
dependencies = ["foo==0.1.0"]

[tool.uv.sources]
foo = { workspace = true }

[tool.uv.workspace]
members = ["packages/foo"]
```

Workspace members are always editable installations.

### Platform-Specific Sources

Limit sources to specific platforms using environment markers:

```toml
[project]
dependencies = ["httpx"]

[tool.uv.sources]
httpx = { git = "https://github.com/encode/httpx", tag = "0.27.2", marker = "sys_platform == 'darwin'" }
```

uv includes the dependency on all platforms but downloads from GitHub on macOS, falling back to PyPI elsewhere.

### Multiple Sources

Specify multiple sources for a single dependency using PEP 508-compatible environment markers:

```toml
[project]
dependencies = ["httpx"]

[tool.uv.sources]
httpx = [
  { git = "https://github.com/encode/httpx", tag = "0.27.2", marker = "sys_platform == 'darwin'" },
  { git = "https://github.com/encode/httpx", tag = "0.24.1", marker = "sys_platform == 'linux'" },
]
```

Platform-specific indexes example:

```toml
[project]
dependencies = ["torch"]

[tool.uv.sources]
torch = [
  { index = "torch-cpu", marker = "platform_system == 'Darwin'"},
  { index = "torch-gpu", marker = "platform_system == 'Linux'"},
]

[[tool.uv.index]]
name = "torch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "torch-gpu"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

### Disabling Sources

Ignore `tool.uv.sources` to simulate resolution with published metadata:

```bash
$ uv lock --no-sources
```

This also prevents workspace member discovery.

---

## Optional and Development Dependencies

### Optional Dependencies

Projects published as libraries often use optional dependencies ("extras") to reduce default dependency trees. Extras are requested with `package[<extra>]` syntax, e.g., `pandas[plot, excel]`.

Define in `[project.optional-dependencies]`:

```toml
[project]
name = "pandas"
version = "1.0.0"

[project.optional-dependencies]
plot = ["matplotlib>=3.6.3"]
excel = [
  "odfpy>=1.4.1",
  "openpyxl>=3.1.0",
  "python-calamine>=0.1.7",
  "pyxlsb>=1.0.10",
  "xlrd>=2.0.1",
  "xlsxwriter>=3.0.5"
]
```

Add optional dependencies with `--optional`:

```bash
$ uv add httpx --optional network
```

### Development Dependencies

Development dependencies are local-only and excluded from publication to PyPI. They don't appear in the `[project]` table.

Add with `--dev`:

```bash
$ uv add --dev pytest
```

Creates a `dev` group:

```toml
[dependency-groups]
dev = ["pytest >=8.1.1,<9"]
```

The `dev` group is special-cased with `--dev`, `--only-dev`, and `--no-dev` flags, and syncs by default.

### Dependency Groups

Divide development dependencies into multiple groups using `--group`:

```bash
$ uv add --group lint ruff
```

Results in:

```toml
[dependency-groups]
dev = ["pytest"]
lint = ["ruff"]
```

Use `--all-groups`, `--no-default-groups`, `--group`, `--only-group`, and `--no-group` to include or exclude groups.

**Tip:** The `--dev`, `--only-dev`, and `--no-dev` flags are shortcuts for `--group dev`, `--only-group dev`, and `--no-group dev`.

uv requires all dependency groups to be compatible, resolving them together when creating lockfiles. Incompatible groups cause resolution to fail.

### Nesting Groups

Groups can include other groups:

```toml
[dependency-groups]
dev = [
  {include-group = "lint"},
  {include-group = "test"}
]
lint = ["ruff"]
test = ["pytest"]
```

### Default Groups

By default, uv includes the `dev` group during `uv run` or `uv sync`. Customize with `tool.uv.default-groups`:

```toml
[tool.uv]
default-groups = ["dev", "foo"]
```

Enable all groups:

```toml
[tool.uv]
default-groups = "all"
```

### Build Dependencies

Projects structured as Python packages declare build-time-only dependencies in `[build-system]` under `build-system.requires`, per PEP 518:

```toml
[project]
name = "pandas"
version = "0.1.0"

[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"
```

### Editable Dependencies

Regular installations build wheels and copy source files. Editable installations add a link (`.pth` file) to source files, keeping environments current with changes.

Add editable dependencies:

```bash
$ uv add --editable ./path/foo
```

Opt out:

```bash
$ uv add --no-editable ./path/foo
```

### Virtual Dependencies

uv allows "virtual" dependencies where the dependency isn't installed as a package, but its dependencies are.

By default, dependencies are never virtual. A `path` dependency becomes virtual with `package = false`:

```toml
[project]
dependencies = ["bar"]

[tool.uv.sources]
bar = { path = "../projects/bar", package = false }
```

Override by setting `package = true`:

```toml
[project]
dependencies = ["bar"]

[tool.uv.sources]
bar = { path = "../projects/bar", package = true }
```

Workspace members can be virtual similarly. Non-dependent workspace members are virtual by default if lacking a build system:

```toml
[project]
name = "parent"
version = "1.0.0"
dependencies = []

[tool.uv.workspace]
members = ["child"]
```

With child lacking a build system, `child` isn't installed but its transitive dependencies are. Declaring dependency changes this:

```toml
[project]
name = "parent"
version = "1.0.0"
dependencies = ["child"]

[tool.uv.sources]
child = { workspace = true }

[tool.uv.workspace]
members = ["child"]
```

Now `child` builds and installs.

### Dependency Specifiers

uv uses standard dependency specifiers per PEP 508, composed of:

- Dependency name
- Extras (optional)
- Version specifier
- Environment marker (optional)

Version specifiers are comma-separated and cumulative: `foo >=1.2.3,<2,!=1.4.0` means "at least 1.2.3, less than 2, not 1.4.0". Specifiers pad with trailing zeros, so `foo ==2` matches 2.0.0.

Stars work with equals for final digits: `foo ==2.1.*` accepts any 2.1 release. The `~=` operator matches where the final digit is equal or higher: `foo ~=1.2` equals `foo >=1.2,<2`.

Extras are comma-separated in square brackets: `pandas[excel,plot] ==2.2`.

Platform-specific requirements use environment markers:
- `importlib-metadata >=7.1.0,<8; python_version < '3.10'`
- `colorama >=0.4.6,<5; platform_system == "Windows"`

Combine markers with `and`, `or`, and parentheses.

---

## Locking and Syncing

### Overview

Locking is the process of resolving your project's dependencies into a lockfile. Syncing is the process of installing a subset of packages from the lockfile into the project environment.

### Automatic Lock and Sync

The locking and syncing processes occur automatically in uv workflows. When executing `uv run`, the project is locked and synced before running the requested command, ensuring the environment stays current.

#### Disabling Automatic Behavior

Prevent automatic lockfile updates using the `--locked` flag:
```bash
$ uv run --locked ...
```

Use the `--frozen` option to rely on an existing lockfile without verification:
```bash
$ uv run --frozen ...
```

Skip environment synchronization with the `--no-sync` option:
```bash
$ uv run --no-sync ...
```

### Checking the Lockfile

Verify lockfile currency by running:
```bash
$ uv lock --check
```

uv will not consider lockfiles outdated when new versions of packages are released — the lockfile needs to be explicitly updated if you want to upgrade dependencies.

### Creating the Lockfile

Generate or update the lockfile explicitly:
```bash
$ uv lock
```

### Syncing the Environment

Manually synchronize the environment:
```bash
$ uv sync
```

This proves useful for ensuring your development tools have correct dependency versions.

#### Editable Installation

By default, projects and workspace members install as editable packages, so re-syncing isn't necessary after code modifications. Disable this behavior with `--no-editable`.

#### Retaining Extraneous Packages

Syncing removes unspecified packages by default. Preserve extra packages using:
```bash
$ uv sync --inexact
```

#### Syncing Optional Dependencies

Include extras from `[project.optional-dependencies]`:
```bash
$ uv sync --extra foo
```

Enable all extras with `--all-extras`.

#### Syncing Development Dependencies

Development dependencies from `[dependency-groups]` (PEP 735) sync automatically. The `dev` group receives special treatment as a default.

Exclude development dependencies:
```bash
$ uv sync --no-dev
```

Install only development dependencies:
```bash
$ uv sync --only-dev
```

Additional group management options include `--all-groups`, `--no-default-groups`, `--group <name>`, `--only-group <name>`, and `--no-group <name>`.

### Upgrading Locked Package Versions

Upgrade all packages to their latest compatible versions:
```bash
$ uv lock --upgrade
```

Upgrade a single package:
```bash
$ uv lock --upgrade-package <package>
```

Upgrade to a specific version:
```bash
$ uv lock --upgrade-package <package>==<version>
```

### Partial Installations

For optimized Docker layering, install dependencies in stages:

- `--no-install-project`: Exclude the current project
- `--no-install-workspace`: Exclude all workspace members
- `--no-install-package <name>`: Exclude specific packages

---

## Running Commands

### Overview

When developing a project, uv installs dependencies into an isolated virtual environment at `.venv`. To execute commands requiring the project environment, use `uv run`:

```bash
$ uv run python -c "import example"
```

The `uv run` command automatically ensures the project environment is current before executing your command.

Commands can originate from the project environment or external sources:

```bash
$ # Running a project-provided CLI tool
$ uv run example-cli foo

$ # Running bash scripts that need project access
$ uv run bash scripts/foo.sh
```

### Requesting Additional Dependencies

You can specify extra dependencies or alternate versions for individual invocations using the `--with` flag:

```bash
$ uv run --with httpx==0.26.0 python -c "import httpx; print(httpx.__version__)"
0.26.0

$ uv run --with httpx==0.25.0 python -c "import httpx; print(httpx.__version__)"
0.25.0
```

Requested versions take precedence over project requirements.

### Running Scripts

Scripts with inline metadata execute in isolated environments separate from your project. Example script:

```python
# /// script
# dependencies = [
#   "httpx",
# ]
# ///

import httpx

resp = httpx.get("https://peps.python.org/api/peps.json")
data = resp.json()
print([(k, v["title"]) for k, v in data.items()][:10])
```

Running `uv run example.py` executes only with specified dependencies.

---

## Resolution

### Overview

Resolution is the process of converting a list of requirements into a list of package versions that satisfy those requirements. This requires recursively searching for compatible versions while ensuring all declared dependencies are compatible.

### Dependencies

Projects typically have dependencies—other packages necessary for functionality. These fall into two categories:

- **Direct dependencies**: packages explicitly declared by the current project
- **Indirect/transitive dependencies**: dependencies added by the project's dependencies

### Platform Markers

Markers attach conditions to requirements, indicating when dependencies apply. For example: `bar ; python_version < "3.9"` installs `bar` only on Python 3.8 and earlier.

Markers are important for resolution because they change required dependencies based on the target platform.

### Universal Resolution

The `uv.lock` file uses universal resolution and is portable across platforms, ensuring dependencies remain consistent regardless of operating system, architecture, or Python version.

During universal resolution:
- Packages may appear multiple times with different versions for different platforms
- All packages must be compatible with the entire `requires-python` range declared in `pyproject.toml`
- Only lower bounds of `requires-python` are considered; upper bounds are ignored

### Limited Resolution Environments

Constrain the set of solved platforms using the `environments` setting:

```toml
[tool.uv]
environments = [
    "sys_platform == 'darwin'",
    "sys_platform == 'linux'",
]
```

### Resolution Strategy

By default, uv uses the latest version of each package. Alternative strategies:

- `--resolution lowest`: Uses the lowest possible version for all dependencies
- `--resolution lowest-direct`: Uses lowest versions for direct dependencies, latest for transitive ones

For libraries, test with `--resolution lowest` or `--resolution lowest-direct` in CI to validate lower bound compatibility.

### Pre-release Handling

By default, uv accepts pre-releases only when:

1. The package is a direct dependency with a pre-release specifier (e.g., `flask>=2.0.0rc1`)
2. All published versions are pre-releases

### Multi-Version Resolution

During universal resolution, packages may appear multiple times with different versions for different platforms. The `--fork-strategy` setting controls the tradeoff:

- `--fork-strategy requires-python` (default): Optimizes for selecting the latest version per Python version while minimizing total versions
- `--fork-strategy fewest`: Minimizes the number of versions, preferring older versions compatible with wider ranges

### Dependency Constraints

Constraint files (`--constraint constraints.txt`) narrow acceptable versions without adding packages to the resolution:

```bash
uv pip compile --constraint constraints.txt requirements.in
```

### Dependency Overrides

Overrides bypass unsuccessful resolutions by replacing declared dependencies:

```toml
[tool.uv]
override-dependencies = ["pydantic>=1.0,<3"]
```

### Dependency Metadata

Provide static metadata upfront using `tool.uv.dependency-metadata`:

```toml
[[tool.uv.dependency-metadata]]
name = "chumpy"
version = "0.70"
requires-dist = ["numpy>=1.8.1", "scipy>=0.13.0", "six>=1.11.0"]
```

### Conflicting Dependencies

uv requires all project dependencies to be compatible. Declare conflicts explicitly:

```toml
[tool.uv]
conflicts = [
    [
        { extra = "extra1" },
        { extra = "extra2" },
    ],
]
```

### Reproducible Resolutions

Use `--exclude-newer` to limit resolution to distributions published before a specific date:

```bash
uv pip compile --exclude-newer 2006-12-02T02:07:43Z requirements.in
```

---

## Exporting Lockfiles

### Overview

uv enables exporting lockfiles to multiple formats for integration with different tools and workflows.

### Available Export Formats

uv provides three export options:

1. **requirements.txt** - Traditional pip-compatible format
2. **pylock.toml** - Standardized Python lockfile format (PEP 751)
3. **CycloneDX** - Industry-standard Software Bill of Materials (SBOM) format

### Specifying Format

Use the `--format` flag to select output:

```bash
$ uv export --format requirements.txt
$ uv export --format pylock.toml
$ uv export --format cyclonedx1.5
```

### Writing to Files

By default, `uv export` outputs to stdout. Use `--output-file` to save:

```bash
$ uv export --format requirements.txt --output-file requirements.txt
$ uv export --format pylock.toml --output-file pylock.toml
$ uv export --format cyclonedx1.5 --output-file sbom.json
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Add optional dependency | `uv add --optional <extra> <package>` |
| Remove dependency | `uv remove <package>` |
| Upgrade all packages | `uv lock --upgrade` |
| Upgrade single package | `uv lock --upgrade-package <package>` |
| Sync environment | `uv sync` |
| Sync with extras | `uv sync --extra <name>` |
| Check lockfile | `uv lock --check` |
| Run command | `uv run <command>` |
| Run with extra deps | `uv run --with <package> <command>` |
| Export to requirements.txt | `uv export --format requirements.txt` |
