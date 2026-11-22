"""Paper sources. Only the collector may use them; nothing else in paperwatch contacts the internet.

Network sources must fetch through paperwatch.network.provider_gateway (fixed operations, read-only rules, budgets,
redacted logging); a source module never builds a URL or opens a connection itself.

So far only the recorded-fixture source exists. arXiv, OpenAlex and Zotero sources implement the same
`PaperSource` protocol from `paperwatch.sources.types`.
"""
