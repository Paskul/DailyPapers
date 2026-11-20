# Operator settings

Edit your topics, provider keys, and configuration here, then install them into the paperwatch home:

```bash
deploy/apply_settings.sh                 # default home: /srv/paperwatch (pass another path as the first argument)
```

| File | Committed | Installed as | Access |
| --- | --- | --- | --- |
| `topics.toml` | no (private questions) | `<home>/topics.toml` | you + group `paperwatch-state`, read-only for the services |
| `config.toml` | no | `<home>/config.toml` | same |
| `collector_credentials.toml` | no (API keys) | `<home>/credentials/collector.toml` | `paperwatch-collector` only, mode 0600 |
| `seeds/*.json` | no | `<home>/seeds/` | same as topics; referenced as `csl_json_files = ["seeds/<file>.json"]` |
| `*.example.toml` | yes | — | starting points to copy |

`config.example.toml` is a link to the package's built-in template (`src/paperwatch/templates/config.toml`), so it
always shows the current defaults. Copy it; don't edit it.

The script first validates every file with the repository's code (unknown keys, bad collection keys, credentials file
permissions, CSL files that don't exist). It changes nothing if validation fails. It skips files that don't exist
here, and asks for `sudo` only to install the credentials file or create `<home>/seeds/`.

To bring an existing live credentials file into this folder (it stays private: mode 0600, gitignored):

```bash
(umask 077; sudo cat /srv/paperwatch/credentials/collector.toml > settings/collector_credentials.toml)
```
