"""
STRUCTURED LITERATURE SEARCH, 2025-2026: LIDC-IDRI deep-learning malignancy classifiers.

The Related Work tallies how studies declare their partition unit. A purposive sample can illustrate
that but cannot support a rate, so this search makes the selection reproducible: the same three
queries, run on a recorded date, deduplicated by DOI (title when there is none). Screening and
extraction happen afterwards, by reading, and are recorded per study in partition_unit_survey.md and
outputs/literature/; this script only retrieves and deduplicates.

Sources: PubMed (E-utilities), Europe PMC (REST) and OpenAlex (which also reaches IEEE Xplore and
most conference proceedings). Scopus and Web of Science were not available.

The published table keeps identifiers, titles and venues. Abstracts, which screening read, belong to
their publishers and are not redistributed: they go, with the raw API responses, to local/, which
is not part of the repository.

    python scripts/literature_search.py            # writes outputs/literature/search_<date>/

The APIs ask callers to identify themselves with a contact e-mail (NCBI's `email`, OpenAlex's
`mailto`). Set it in the environment rather than in this file:

    set LITSEARCH_MAILTO=you@example.org           # Windows; `export` on other shells

Without it the queries are sent anonymously, which the APIs accept at lower rate limits.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

MAILTO = os.environ.get("LITSEARCH_MAILTO", "").strip()   # contact e-mail for the APIs, optional
UA = {"User-Agent": "nodule-partition-leakage literature search"
                    + (f" (mailto:{MAILTO})" if MAILTO else "")}
_ID = {"email": MAILTO} if MAILTO else {}      # NCBI E-utilities
_OA = {"mailto": MAILTO} if MAILTO else {}     # OpenAlex

PUBMED = ('("LIDC-IDRI"[tiab] OR "LIDC"[tiab] OR "Lung Image Database Consortium"[tiab] OR "LUNA16"[tiab]) '
          'AND (malignan*[tiab] OR benign*[tiab]) '
          'AND ("deep learning"[tiab] OR convolutional[tiab] OR "neural network"[tiab] OR '
          '"neural networks"[tiab] OR transformer*[tiab] OR CNN[tiab]) '
          'AND ("2025/01/01"[dp] : "3000"[dp])')


def _ta(term):
    return f'(TITLE:{term} OR ABSTRACT:{term})'


EUROPEPMC = (f'({_ta(chr(34) + "LIDC-IDRI" + chr(34))} OR {_ta("LIDC")} OR '
             f'{_ta(chr(34) + "Lung Image Database Consortium" + chr(34))} OR {_ta("LUNA16")}) '
             f'AND ({_ta("malignan*")} OR {_ta("benign*")}) '
             f'AND ({_ta(chr(34) + "deep learning" + chr(34))} OR {_ta("convolutional")} OR '
             f'{_ta(chr(34) + "neural network" + chr(34))} OR {_ta(chr(34) + "neural networks" + chr(34))} OR '
             f'{_ta("transformer*")} OR {_ta("CNN")}) '
             'AND (PUB_YEAR:[2025 TO 2026]) NOT (SRC:PPR)')
OPENALEX_FILTER = "publication_year:2025-2026,type:article|proceedings-article"
OPENALEX = ('(LIDC OR "LIDC-IDRI" OR "Lung Image Database Consortium" OR LUNA16) AND '
            '(malignant OR malignancy OR benign) AND '
            '("deep learning" OR convolutional OR "neural network" OR transformer OR CNN)')


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:                                   # transient network error: back off
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def search_pubmed():
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    q = urllib.parse.urlencode({"db": "pubmed", "term": PUBMED, "retmax": 5000, "retmode": "json",
                                **_ID})
    ids = json.loads(get(base + "esearch.fcgi?" + q))["esearchresult"]["idlist"]
    recs = []
    for i in range(0, len(ids), 200):
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids[i:i + 200]), "retmode": "xml",
                                    **_ID})
        root = ET.fromstring(get(base + "efetch.fcgi?" + q))
        for art in root.iter("PubmedArticle"):
            m = art.find("MedlineCitation"); a = m.find("Article")
            # The record's OWN identifiers only: art.iter() would also walk PubmedData/ReferenceList
            # and could hand a record the DOI of a paper it cites.
            ids = art.find("PubmedData/ArticleIdList")
            ids = list(ids) if ids is not None else []
            doi = next((x.text for x in ids if x.get("IdType") == "doi"), None)
            pmc = next((x.text for x in ids if x.get("IdType") == "pmc"), None)
            abstract = " ".join("".join(x.itertext()) for x in a.iter("AbstractText"))
            year = (a.findtext("Journal/JournalIssue/PubDate/Year") or
                    (a.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "")[:4])
            types = [x.text for x in a.iter("PublicationType")]
            recs.append(dict(source="pubmed", id=m.findtext("PMID"), doi=(doi or "").lower() or None,
                             pmcid=pmc, title="".join(a.find("ArticleTitle").itertext()),
                             year=year, venue=a.findtext("Journal/Title"), types="; ".join(types),
                             abstract=abstract))
        time.sleep(0.4)
    return recs


def search_europepmc():
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
    recs, cursor = [], "*"
    while True:
        q = urllib.parse.urlencode({"query": EUROPEPMC, "format": "json", "pageSize": 1000,
                                    "resultType": "core", "cursorMark": cursor})
        d = json.loads(get(base + q))
        for r in d.get("resultList", {}).get("result", []):
            recs.append(dict(source="europepmc", id=r.get("id"), doi=(r.get("doi") or "").lower() or None,
                             pmcid=r.get("pmcid"), title=r.get("title"), year=r.get("pubYear"),
                             venue=(r.get("journalInfo") or {}).get("journal", {}).get("title")
                             or r.get("bookOrReportDetails", {}).get("publisher"),
                             types="; ".join((r.get("pubTypeList") or {}).get("pubType", [])),
                             abstract=r.get("abstractText") or ""))
        nxt = d.get("nextCursorMark")
        if not nxt or nxt == cursor or not d.get("resultList", {}).get("result"):
            break
        cursor = nxt
    return recs


def _abstract_oa(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def search_openalex():
    base = "https://api.openalex.org/works?"
    recs, cursor = [], "*"
    filt = f"{OPENALEX_FILTER},title_and_abstract.search:{OPENALEX}"
    while cursor:
        q = urllib.parse.urlencode({"filter": filt, "per-page": 200, "cursor": cursor, **_OA})
        d = json.loads(get(base + q))
        for w in d.get("results", []):
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            oa = (w.get("open_access") or {}).get("oa_url")
            recs.append(dict(source="openalex", id=w.get("id"),
                             doi=(w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
                             pmcid=(w.get("ids") or {}).get("pmcid"), title=w.get("title"),
                             year=w.get("publication_year"), venue=src.get("display_name"),
                             types=f"{w.get('type')}; source:{src.get('type')}", abstract=_abstract_oa(
                                 w.get("abstract_inverted_index")), oa_url=oa))
        cursor = (d.get("meta") or {}).get("next_cursor")
        time.sleep(0.2)
    return recs


def main():
    today = datetime.date.today().isoformat()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "outputs", "literature", f"search_{today}")
    local = os.path.join(out, "local")                      # abstracts and raw responses, not published
    os.makedirs(local, exist_ok=True)
    found = {}
    for name, fn in (("pubmed", search_pubmed), ("europepmc", search_europepmc), ("openalex", search_openalex)):
        recs = fn()
        found[name] = recs
        json.dump(recs, open(os.path.join(local, f"raw_{name}.json"), "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)
        print(f"{name:10s} {len(recs):5d} records")

    merged = {}
    for name in ("pubmed", "europepmc", "openalex"):
        for r in found[name]:
            key = r["doi"] or "title:" + norm_title(r["title"])
            m = merged.setdefault(key, dict(key=key, doi=r["doi"], title=r["title"], year=r["year"],
                                            venue=r["venue"], types=r["types"], pmcid=r.get("pmcid"),
                                            oa_url=r.get("oa_url"), abstract=r["abstract"], sources=[]))
            m["sources"].append(name)
            for f in ("pmcid", "oa_url", "venue"):
                m[f] = m[f] or r.get(f)
            if len(r["abstract"] or "") > len(m["abstract"] or ""):
                m["abstract"] = r["abstract"]
    # a record without a DOI may duplicate one with a DOI: fold it in by title
    by_title = {norm_title(m["title"]): k for k, m in merged.items() if m["doi"]}
    for k in [k for k, m in merged.items() if not m["doi"]]:
        t = norm_title(merged[k]["title"])
        if t in by_title:
            merged[by_title[t]]["sources"] += merged.pop(k)["sources"]
    rows = sorted(merged.values(), key=lambda m: (str(m["year"]), norm_title(m["title"])))
    cols = ["key", "doi", "title", "year", "venue", "types", "sources", "pmcid", "oa_url"]
    for path, extra in ((os.path.join(out, "records_deduplicated.csv"), []),
                        (os.path.join(local, "records_with_abstracts.csv"), ["abstract"])):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols + extra)
            for m in rows:
                w.writerow([m["key"], m["doi"], m["title"], m["year"], m["venue"], m["types"],
                            "+".join(sorted(set(m["sources"]))), m["pmcid"], m["oa_url"]]
                           + [m[c] for c in extra])
    log = dict(date=today, queries=dict(pubmed=PUBMED, europepmc=EUROPEPMC, openalex=OPENALEX,
                                        openalex_filter=OPENALEX_FILTER),
               counts={**{k: len(v) for k, v in found.items()}, "deduplicated": len(rows)})
    json.dump(log, open(os.path.join(out, "search_log.json"), "w", encoding="utf-8"), indent=1)
    print(f"deduplicated {len(rows)} records -> {out}")


if __name__ == "__main__":
    main()
