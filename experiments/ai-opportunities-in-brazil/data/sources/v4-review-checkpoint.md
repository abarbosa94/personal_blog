# V4 affiliation-resolution checkpoint

Date: 2026-07-28.

This checkpoint freezes the blind review of the 35 records left out of the
previous V3 tuning sample. All 35 decisions were completed before the V4 rules
were added.

## Frozen review

| Measure | Value |
| --- | ---: |
| Reviewed records | 35 |
| Pass | 29 |
| Fail | 6 |
| Defer | 0 |

The frozen queue SHA-256 is
`c8928816d967c9cfbb27d8c27ed7fa6fe0ad69eed152ec72a646d3adc479b99b`.

## Reviewed failure mechanisms

- explicit country or city evidence was not converted to an ISO country;
- a sole company affiliation such as Salesforce was not retained;
- a generic unit name was overmatched to the wrong ROR organization;
- an unqualified multinational lab name was assigned to the wrong branch;
- the review expectation parser counted rejected pipeline countries appearing
  later in explanatory notes.

## V4 changes

- Added reviewed, conservative country/location aliases for Colombia, Austria,
  and high-signal US cities found in affiliation addresses.
- Added narrow organization aliases for Salesforce, CeRAI/IIT Madras, and
  WSAI/IIT Madras.
- Rejected bare `Google DeepMind` as branch-ambiguous.
- Rejected `Institute for Machine Learning` when ROR returns that generic unit
  for a richer affiliation string.
- Made the expectation parser use the first explicit expected-country statement
  instead of all country mentions in a review note.

## Six-case gate

| Measure | Reviewed V3 | V4 |
| --- | ---: | ---: |
| Exact country sets | 0/6 | 5/6 |
| Country recall | 22.2% | 88.9% |
| Cases with an extra country | 2 | 0 |

The remaining non-exact case is `neurips:61432775ec681f7ac6e7d335fa933e4f`
(`InvisibleInk`). V4 recovers India and removes the incorrect United Kingdom,
but the paper prints only `Google DeepMind`; determining the author's US branch
requires evidence outside the affiliation string. This case should remain
reviewable rather than receive a global company-country mapping.

When V4 was run across all 35 records, country sets changed only for the six
reviewed failures. None of the 29 reviewed passes changed.

## Test result

The complete local suite passes: `42 passed`.

## SHA-256

- frozen failure expectations:
  `3dc0311a1494bb11d21b369508fefcb25dd4265f728176f17f7f99d82664ed9d`;
- V4 six-case output:
  `a170afde76e3f413bb48c604765032e2c8819db2c8cfccbc397a2fadcae7e211`;
- V4 35-record output:
  `22dc4b2f40179251e72f9c7b970d204aec9cad226dd9cd20facd253d4ea98d09`;
- detailed comparison:
  `1084cff71d297cf6de9d07318de0b497cfd29d933f497e4eb81235843f833871`.

