# Data

The benchmark uses **105 datasets** (listed in `../datasets_105.txt`).

- **Bundled here:** 5 small samples (`misc/`, `config/`, `process/`) for a quick
  start, plus 2 `hpo/` datasets that are not in MOOT.
- **Fetched:** the remaining 103 come from the public **MOOT** suite
  (<https://github.com/timm/moot>). Run `python ../fetch_data.py` to clone MOOT
  and copy them into `data/<category>/`.

Column convention (from `ezr`): names ending in `+`/`-` are objectives to
maximize/minimize, uppercase-initial names are numeric, the rest are features.
