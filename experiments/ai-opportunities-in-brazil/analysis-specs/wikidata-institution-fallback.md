# Wikidata fallback for institution geography

## Scope

Wikidata is a tertiary fallback for organization-level analysis when an
institution has been identified but explicit affiliation geography and a
chosen ROR match are unavailable. It does not alter the completed seven-venue
country census unless a new version of that indicator is explicitly opened.

## Evidence precedence

1. country explicitly stated in the paper affiliation;
2. chosen ROR organization and its country;
3. Wikidata headquarters location (`P159` → `P17`);
4. country of origin (`P495`);
5. formation place (`P740` → `P17`);
6. direct country (`P17`).

The selected property and Wikidata QID must be retained as provenance.
Conflicting properties are not collapsed silently. A headquarters/origin
country describes the organization and must not be presented as the location
of an individual author or branch office.

## Entity resolution and review

Automatic country lookup is allowed only for a known Wikidata QID. Name search
produces candidates, not accepted matches. Ambiguous names require type,
official website/domain, aliases, and parent-organization evidence or manual
review. Queries should batch QIDs with `VALUES`, use a descriptive User-Agent,
cache responses, respect HTTP 429/`Retry-After`, and use bounded exponential
backoff.

Spotify (`Q689141`) is a validation fixture: the current record maps to Sweden
through country of origin and formation place; it currently does not return a
headquarters `P159` result.
