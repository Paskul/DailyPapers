"""Follow the important references of completed reviews (plan part 8).

    reference_fetch.py           collector stage: a reviewed paper's OpenAlex reference list, plus identifiers from
                                 its own bibliography confirmed by a provider record; cited works enter the catalog
                                 and the same screening as every other candidate
    bibliography_identifiers.py  find DOIs / arXiv ids in bibliography passages; confirm a provider record's title
    citation_hops.py             which reviewed papers are followed, and how many hops from a screened paper they are
    reference_candidates.py      the cited works offered to the research role, with their citation contexts
    citation_contexts.py         where the parent paper's text cites a work ("[12]", "Smith et al., 2019")
    research_branches.py         unused approved searches (query + year window) offered for a gap; run by `collect`
    reference_selection.py       queue the research role's job; turn its validated choices into class-3 readings
                                 within the daily limit (reference_admission.py)
The model never composes a request: it names a candidate id or a branch id from the list it was given, and host code
checks the id, the hop limit, the daily limits, and cycles before anything is queued or searched.
"""
