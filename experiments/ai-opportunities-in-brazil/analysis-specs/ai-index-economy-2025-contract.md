# Frozen contract — AI Index/Quid economic indicators, 2025

The two indicators are official aggregates published in the 2026 AI Index
Economy chapter using Quid 2025 data:

- Figure 4.2.8: total private AI investment, billions of current US dollars;
- Figure 4.2.9: number of newly funded AI companies.

The source publishes only the 15 leading geographies. A panel country absent
from a figure is stored as `not_reported_top15`, not zero. Its only quantitative
claim is an upper bound equal to the smallest published top-15 value. Ranks are
computed only among panel countries with published values.

Private investment does not include public funding, government guidance funds,
or all forms of corporate capital expenditure. Company geography follows
Quid's published classification and is not independently reconstructed.

Stanford's `fig_4.2.9.csv` contains the China–Europe–US time series rather than
the top-15 cross-section shown in Figure 4.2.9. The cross-section is therefore
extracted from Stanford's official chart PDF and checked against the chapter
text. This source inconsistency is retained in metadata and documentation.
