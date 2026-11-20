# Operator settings

Everything you set lives in this folder, and `./setup.sh` (at the repository root) applies it:

```bash
./setup.sh                  # install / update everything settings/paperwatch.toml describes (safe to re-run)
./setup.sh --settings-only  # quick path after editing topics or config: settings, schedule, monitor only
./setup.sh --check          # validate; change nothing
```

| File | Committed | What it holds | Installed as |
| --- | --- | --- | --- |
| `paperwatch.toml` | no (private questions) | `[setup]` (home, mode, schedule, monitor), every config table, and your `[[topics]]` | `<home>/config.toml` + `<home>/topics.toml` |
| `collector_credentials.toml` | no (API keys) | Zotero and OpenAlex keys, mode 0600 | `<home>/credentials/collector.toml`, readable by the collector only |
| `seeds/*.json` | no | CSL JSON seed exports, listed in a topic as `csl_json_files = ["seeds/<file>.json"]` | `<home>/seeds/` |
| `.extractor-image-id` | no | the id of the PDF extractor image setup built (used when `[extractor] image` is empty) | — |
| `*.example.toml` | yes | starting points | — |

The first `./setup.sh` creates `paperwatch.toml` for you: from the example, or, if you still have the former
`config.toml` + `topics.toml` here, by combining them unchanged (they are then no longer read; delete them when you
are happy with the new file). It stops so you can review the `[setup]` table before anything is installed.

`paperwatch.example.toml` is generated from the package's templates, so it always shows every option with its
current default (a test keeps it in step). Copy it; don't edit it.

Setup first validates the whole file with the application's own loader (unknown keys, bad values, cross-field
rules), checks that every CSL JSON seed file exists and that Zotero topics have a key, and proves that the file
splits exactly into `config.toml` and `topics.toml`. Nothing is installed if any check fails.

To bring an existing live credentials file into this folder (it stays private: mode 0600, gitignored):

```bash
(umask 077; sudo cat /srv/paperwatch/credentials/collector.toml > settings/collector_credentials.toml)
```
