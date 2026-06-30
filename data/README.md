# Data

The benchmark uses **105 datasets** (listed in `../datasets_105.txt`).

- **Bundled here:** 7 datasets (5 in `misc/`, `config/`, `process/` and 2 in
  `hpo/`), all from MOOT, kept locally for a quick start.
- **Fetched:** the rest come from the public **MOOT** suite
  (<https://github.com/timm/moot>). Run `python ../fetch_data.py` to clone MOOT
  and copy them into `data/<category>/`.

Column convention (from `ezr`): names ending in `+`/`-` are objectives to
maximize/minimize, uppercase-initial names are numeric, the rest are features.
