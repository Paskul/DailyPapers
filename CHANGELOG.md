# Changelog

Notable changes, generated from the Conventional Commit messages with [git-cliff](https://git-cliff.org).
Versions follow [Semantic Versioning](https://semver.org); before 1.0.0, a minor version may change behavior.

## [Unreleased]

### Features

- llm: Record llama.cpp's own timings for every model call

### Bug fixes

- Resolve the findings of ruff, ShellCheck, and Biome
- screening: Classify a full-timestamp publication date by its day

### Performance

- cli: Import each command's code only when that command runs
- Run numpy with one BLAS thread unless the operator sets another

### Refactoring

- types: Pass mypy --strict in src and type-check the tests
- Describe the code itself instead of the internal build plan

### Documentation

- Add contributing guide, security policy, and issue templates
- Add performance measurements

### Tests

- Guard the monitor page against HTML injection; set a coverage floor

### Build and CI

- Add ruff, mypy, Biome, and pre-commit configuration
- Run the checks, the tests, and the real extractor sandbox on GitHub Actions
- Add CodeQL scanning and Dependabot updates
- Publish GitHub releases with notes from the commit history

## Before 0.1.0

The project was built before it used Conventional Commits; its earlier history is in `git log`.
