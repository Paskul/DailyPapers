"""Seed papers from the user's own library (plan part 4).

A seed is one bibliographic record from a topic's Zotero collection (live Web API) or CSL JSON export (offline file).
Both sources have equal standing: same record type, same tables, same role in relevance.

    seed_items.py      SeedItem (one parsed seed), seed ids, discovery routes, identifier extraction
    zotero_items.py    parse one page of Zotero Web API items (drops notes and attachments)
    csl_json_items.py  parse a CSL JSON export (drops free-text notes)
    seed_store.py      the seed cache in SQLite: seed_items, collection memberships, active seeds per source
    seed_sync.py       sync every configured seed source; commit a source's new state only after a complete read

Seeds stay on the machine. Nothing derived from a seed (title, note, collection name) ever becomes a search term; a
seed's identifier may be looked up at a provider only with collector.allow_lookup_by_seed_identifier = true.
Keep this __init__ free of imports: the network gateway imports seed_items, and must not pull in the database layer.
"""
