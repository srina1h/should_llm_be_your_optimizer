# Prompts

All LLM prompt construction for the reproduction, kept separate from method logic.

| File | Used by | Builds |
|---|---|---|
| `opro_prompts.py` | OPRO, SNAP2 | The OPRO meta-prompt: hard constraints, objective directions, the running `(config, d2h)` trajectory (sorted worst→best), diversification/retry blocks, and the JSON output schema. |
| `synthetic.py` | LLM (BS_LLM), SynthCore | `SYNTHETIC.get_template_markdown()`: shows the best/rest split and asks for 2 better + 2 worse synthetic configurations (plain-text list). |

Every prompt sent to the model is constructed in this folder and is plain,
inline text — nothing is loaded from an external service or hidden template.
