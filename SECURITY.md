# Security policy

paperwatch is built on the assumption that papers are hostile: a PDF can contain text written to manipulate an
AI. Its design, described in the README under "Security and privacy", is that even a fully compromised model
cannot make paperwatch reach the network, run programs, or write outside its folders. A way around any part of
that is a vulnerability, and reports are very welcome.

## Supported versions

paperwatch is pre-1.0 and has one maintainer. Fixes go into `main` and the next release; only the latest release
is supported.

## Reporting a vulnerability

**Please don't open a public issue.** Report privately through GitHub:
[**Report a vulnerability**](https://github.com/Paskul/DailyPapers/security/advisories/new) (the Security tab of
the repository, then "Report a vulnerability").

Please include:

- the version (`paperwatch --version`) or commit, and the setup mode (`local` or `hardened`);
- what an attacker controls (a paper's text, a PDF, a provider response, the model's answers) and what they gain;
- steps to reproduce, ideally a synthetic paper or a fixture.

Please leave out your API keys, your private research questions, and copyrighted paper text. A synthetic example
shows the problem just as well.

What to expect: this is a single-maintainer project, so responses are best effort. You can expect an
acknowledgement within 7 days and an assessment within 30. Fixed issues are published as a GitHub security
advisory, crediting you unless you prefer otherwise. Please allow up to 90 days before disclosing publicly.

## Scope

In scope is anything that lets untrusted input (paper text, a PDF, provider data) or the model's output:

- cause a network request that `network/provider_rules.toml` does not define, or reach a host outside a
  command's egress policy;
- choose a request's target, a file path, or a program to run;
- escape the PDF extractor sandbox, or give the extractor network access;
- write outside the paperwatch home, or change another paper's notes or reports;
- inject active Markdown or HTML into notes, reports, or the monitor page;
- leak API keys, private research questions, or paper text into logs, the outbound request ledger, backups, or
  requests to providers;
- make the monitor page reachable from other hosts, or change state through it.

Also in scope:

- weaknesses in the deploy scripts that undermine the hardened deployment (file permissions, service accounts,
  egress rules);
- weaknesses in how dependencies and the screening model and extractor image are pinned and verified.

Out of scope:

- vulnerabilities in llama.cpp, Podman, poppler, Python, the kernel, or other upstream software (please report
  those upstream, and tell us if the way paperwatch uses them makes them worse);
- setups that remove a control and then observe that it is missing, such as running `llama-server` with tools,
  agent, or MCP features, or skipping the hardened mode;
- an attacker who already controls the operator's account or root;
- the quality of the model's reviews (please open an ordinary issue).

## Keeping a deployment patched

- `git pull` and then `./setup.sh` installs exactly the reviewed, hash-pinned dependency versions in `uv.lock`.
- **Rebuild the extractor image now and then**, monthly for example. poppler parses untrusted PDFs, and its Debian
  base image is resolved only when the image is built, so neither Dependabot nor `git pull` updates it. Delete
  `settings/.extractor-image-id` (and any `[extractor] image` in your settings file), then run `./setup.sh`: it
  builds a fresh image and installs its id.
- Keep llama.cpp up to date, and run `llama-server` without `--tools`, `--agent`, MCP options, or remote model
  downloads (`paperwatch doctor` fails if the server exposes tools).
