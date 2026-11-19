"""Everything that may contact the internet. ONLY the collector (`paperwatch collect`) uses this package.

  provider_rules.toml / .py   read-only hosts, paths, parameters, redirects, budgets, response bounds per provider
  provider_gateway.py         the fixed operations: search, get_work, get_text, list_seed_items
  provider_requests.py        builds each request from validated ids and approved catalog queries
  secure_fetcher.py           one bounded HTTPS hop: strict TLS, public addresses only, no automatic redirects
  outbound_ledger.py          request budgets and spacing per provider; redacted log of every hop
  collector_credentials.py    the collector-only credentials file
  egress_guard.py             per-process audit hook: what each component may resolve, connect to, and launch

The worker, report writer, and model server never import the gateway, the fetcher, or the credentials. The model
receives plain data and has zero tools, so no text from a paper or a model answer can become a request.
"""
