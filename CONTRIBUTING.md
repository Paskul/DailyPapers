# Contributing to paperwatch

paperwatch is a personal, experimental project (see the note at the top of the README), but bug reports, ideas,
and pull requests are welcome. This page covers how to set up, the checks, commit messages, the coding standards,
and releases. For a security problem, please don't open an issue: see [SECURITY.md](SECURITY.md).

## Set up

You need Linux, Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). The tests need no model server, no
network, and no API keys.

```bash
git clone https://github.com/Paskul/DailyPapers.git && cd DailyPapers
uv sync                                               # the package plus the development tools
uv run pre-commit install                             # once per clone: checks each commit and its message
git config blame.ignoreRevsFile .git-blame-ignore-revs   # optional: `git blame` skips the reformatting commit
```

## Checks

| What | Command |
| --- | --- |
| Everything pre-commit checks: ruff (lint + format), Biome (monitor page), ShellCheck, mypy, actionlint and zizmor (workflows), gitleaks (secrets), file hygiene | `uv run pre-commit run --all-files` |
| The offline test suite | `uv run pytest` |
| ... with line and branch coverage (must stay at or above 85%) | `uv run pytest --cov` |
| Types only | `uv run mypy` |

The hook runs these checks on the files of each commit, and CI runs all of them on every push and pull request,
plus the tests on Python 3.12 to 3.14, the PDF sandbox with real Podman, and CodeQL. Optional tests against real
components are listed in the README under "Development".

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org): `type(scope): summary`. The
commit-msg hook checks the format; the release notes and CHANGELOG.md are generated from it.

| Type | For |
| --- | --- |
| `feat` | a new capability (a command, an option, a provider operation) |
| `fix` | a bug fix |
| `perf` | faster or lighter, same behavior |
| `refactor` | a code change that changes no behavior |
| `docs`, `test` | documentation only, tests only |
| `build`, `ci` | packaging and dependencies, GitHub Actions |
| `style`, `chore` | formatting only; maintenance that needs no changelog entry |

- The scope is optional; use the package or area: `collection`, `seeds`, `screening`, `reading`, `references`,
  `comparisons`, `reporting`, `rendering`, `monitor`, `roles`, `llm`, `network`, `deployment`, `setup`, `deps`.
- The summary is imperative, at most 72 characters, without a final period: `fix(reading): accept a number
  written as a word`.
- The body says what changed and **why**, wrapped at about 72 columns. Measurements and the reason for a design
  choice belong here.
- A change that breaks a deployment (a config option renamed, a migration that needs a step by hand) gets a `!`
  after the type and a `BREAKING CHANGE:` footer that says what the operator must do.

## Coding standards

### Style

- **Formatting is automatic.** ruff format (Black style, 120 columns) for Python, Biome for the monitor page.
  Don't format by hand, and don't mix reformatting with other changes.
- **Names say what a thing is**, in full: `remaining_candidate_budget`, not `rem`; `paper_notes_dir`, not `d`.
- **Docstrings state the contract**: what a function promises, what it assumes ("call inside a write
  transaction"), and what it never does ("never makes a network request"). Comments explain *why*, not *what*.
- **Small modules with one purpose**, grouped by pipeline stage (see "Project layout" in the README). Each CLI
  command lives in its own module in `commands/`.
- **Types everywhere.** Every function is annotated, and `src/` passes `mypy --strict`. Avoid `Any` unless the value
  really is unknown (parsed JSON before validation). Use frozen dataclasses for records, `StrEnum` for fixed sets of
  values, and `Protocol` for "anything with these fields".
- **Standard library first.** A new runtime dependency is a supply-chain decision: it joins the hash-pinned
  install of every hardened deployment. Explain why in the pull request. Development tools go in the `dev`
  dependency group.

### Data, time, and settings

- **SQLite:** every write happens inside `db.write_transaction` (short, and never around a network call). Values
  are always bound parameters. Dynamic SQL may interpolate only constants, column names from code, or `?` lists,
  and says so: `# noqa: S608  # only ? placeholders are interpolated`.
- **Schema changes** are a new entry at the end of `SCHEMA_MIGRATIONS` (`db_migrations.py`); `paperwatch init`
  applies it. Never change the statements of a migration that has shipped.
- **Time:** every stored or compared time is timezone-aware UTC from `paperwatch.clock` (`utc_now`,
  `format_utc_timestamp`). Tests control the clock through it.
- **Settings:** a new option gets its default and an explanation in `src/paperwatch/templates/config.toml`. Then
  regenerate the example (a test fails until you do):
  `uv run python -m paperwatch.deployment.settings_file write-example settings/paperwatch.example.toml`.
- **Errors** are `PaperwatchError` subclasses (`errors.py`). Each names the failing component, which sets the exit
  code, and carries an `operator_hint` saying what to do about it.

### Security rules

These are the guarantees in the README's "Security and privacy" section, and the tests check them. A change that
weakens one is a security issue, so say so in the pull request if a change touches any of them.

1. **The model has no tools and chooses only among what it was offered.** Each role's output schema forbids
   unknown fields (`additionalProperties: false`). Every id in an answer is checked against the ids in its input,
   and checked again before it is stored (`only_offered_choices`). A change to a prompt or an output schema bumps
   the role's `prompt_version`.
2. **Requests are built by code.** Every outbound request goes through `ProviderGateway`, using an operation
   defined in `network/provider_rules.toml`: an allow-listed host, fixed parameters, a budget, and a row in the
   outbound ledger with credentials redacted. Nothing else opens a network connection.
3. **Each command runs under one egress policy** (`cli.select_egress_policy_for_command`). A new command joins the
   right set. Only `collect` may reach the internet; `access-check` runs unguarded so that it can prove the
   firewall blocks everyone else.
4. **Untrusted text stays inert.** Paper text, titles, provider data, and model output never appear in a file name
   or a path (use internal ids). They are escaped in Markdown (`rendering/markdown_safety.py`), inserted only as
   text in the monitor page (`textContent`, never `innerHTML`; a test enforces this), and never reach a shell.
5. **PDFs are parsed only inside the extractor sandbox** (`sandboxed_extractor.py`): no network, a read-only
   root, no capabilities, and limits on memory, processes, and time.

### Tests

- **Offline and deterministic.** Use `FakeModelClient` (`llm/fake_client.py`), the fake llama.cpp server, the fake
  paper providers (an HTTP transport), and `build_in_process_app` (`tests/conftest.py`). Nothing may reach the
  internet or a real model unless the test is opted in by an environment variable.
- **A test name states the guarantee:** `test_stale_worker_cannot_overwrite_the_reclaiming_workers_result`.
- **A bug fix comes with a test that failed before the fix.**
- Test files are named by feature: `tests/test_<feature>.py`.

## Pull requests

Keep one topic per pull request. The template's checklist covers the rest: green checks, updated docs and settings
example, no private data, and the security rules.

## Releases

For maintainers. Versions follow [Semantic Versioning](https://semver.org); before 1.0.0, a minor version may
change behavior.

1. Set `__version__` in `src/paperwatch/__init__.py` (the only place the version is written).
2. Update the changelog: `uvx git-cliff@2.14.2 --tag vX.Y.Z --output CHANGELOG.md`. Review it, then commit both
   files as `chore(release): vX.Y.Z`.
3. Tag and push: `git tag -a vX.Y.Z -m "paperwatch X.Y.Z" && git push origin main vX.Y.Z`. The release workflow
   checks that the tag matches the version, then publishes the GitHub release with the notes for that version.
