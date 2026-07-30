# Frozen contract — TOP500 indicators

The source universe is each official TOP500 snapshot's 500 ranked systems.
Official HTML pages are frozen locally because bulk downloads require an
account. Every extraction must contain exactly 500 distinct ranks.

Two indicators are reported by installation country:

- number of listed systems;
- sum of HPL `Rmax`, in PFlop/s.

June and November 2025 remain separate; November is the annual baseline. June
2026 is a snapshot/YTD panel, not a full-year flow. TOP500 measures ranked HPC
systems and HPL performance. It does not by itself measure access, accelerator
suitability, availability for AI training, or nationally controlled compute.
