# Frozen contract — Epoch AI model indicators

## Source and temporal panels

The source is Epoch AI's `all_ai_models.csv`, frozen locally with its retrieval
date and SHA-256. The annual panel includes publication dates from 2025-01-01
through 2025-12-31. The 2026 panel is YTD through the retrieval date and must
not be compared as a completed year.

## Notable models

A record is notable when Epoch's `Notability criteria` field is non-empty.
Country is the country of the producing organization, not the training location
or the location where value is captured. A model receives one full presence
count in each represented country, with duplicate country labels removed.

## Academia–industry composition

`Organization`, `Country (of organization)`, and `Organization categorization`
are treated as parallel fields. Within each model-country observation:

- `industry`: at least one industry organization and no academic organization;
- `academia`: at least one academic organization and no industry organization;
- `mixed`: both industry and academic organizations;
- `other`: neither, including government and research collectives.

The output reports category counts and shares within each country's notable
models. Mixed production remains visible and is not assigned arbitrarily to
academia or industry.

## Missingness and interpretation

Models without a mapped panel country remain in the global eligible
denominator reported in metadata but do not enter a panel country count.
Epoch's dataset is non-exhaustive; zero means no observed record under this
taxonomy and snapshot, not proof that a country produced no relevant model.
