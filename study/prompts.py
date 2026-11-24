SYM_VOCAB_DISPLAY_CAP = 30


def _truncate_vocab(vals):
    if len(vals) <= SYM_VOCAB_DISPLAY_CAP:
        return f"{vals}"
    head = vals[:SYM_VOCAB_DISPLAY_CAP]
    return f"{head} ... and {len(vals) - SYM_VOCAB_DISPLAY_CAP} more values"


def _format_constraints(data):
    """Hard constraints, prominently formatted. Each line names allowed values
    explicitly so the LLM cannot drift to domain priors (e.g. 'Ex' for
    bsmtcond when dataset only has Fa/Gd/Po/TA/Unknown).

    For high-cardinality SYMs the listing is capped to keep the prompt under
    the model's context window. The lenient parser accepts out-of-vocab values,
    so truncation does not invalidate any candidate.
    """
    lines = []
    for c in data.cols.x:
        if hasattr(c, "lo"):
            lines.append(f"- {c.txt} (NUM): MUST be numeric, range [{c.lo}, {c.hi}]")
        else:
            vals = list(c.has.keys()) if getattr(c, "has", None) else []
            if vals:
                lines.append(
                    f"- {c.txt} (SYM): preferably one of {_truncate_vocab(vals)}. "
                    f"({len(vals)} known values total.)"
                )
            else:
                lines.append(f"- {c.txt} (SYM): categorical")
    return "\n".join(lines)


def _format_objectives(data):
    lines = []
    for c in data.cols.y:
        direction = "maximize" if getattr(c, "maximize", True) else "minimize"
        lines.append(f"- {c.txt}: {direction}")
    return "\n".join(lines)


def _format_x_row(data, row):
    parts = []
    for c in data.cols.x:
        v = row[c.at]
        parts.append(f"{c.txt}={v}")
    return ", ".join(parts)


def _format_trajectory(data, sorted_traj_desc):
    lines = ["| configuration | d2h |", "|---|---|"]
    for row, score in sorted_traj_desc:
        lines.append(f"| {_format_x_row(data, row)} | {score:.4f} |")
    return "\n".join(lines)


def _format_collisions(data, collisions):
    if not collisions:
        return ""
    lines = []
    for row, score in collisions:
        lines.append(f"- {_format_x_row(data, row)} (d2h={score:.4f})")
    return (
        "Your previous proposals mapped onto these already-evaluated rows. "
        "Diversify — propose configurations distinct from these:\n" + "\n".join(lines) + "\n"
    )


def _format_feature_schema(data):
    schema = {}
    for c in data.cols.x:
        if hasattr(c, "lo"):
            schema[c.txt] = f"<numeric in [{c.lo}, {c.hi}]>"
        else:
            vals = list(c.has.keys()) if getattr(c, "has", None) else []
            if not vals:
                schema[c.txt] = "<categorical>"
            elif len(vals) <= SYM_VOCAB_DISPLAY_CAP:
                schema[c.txt] = f"<one of {vals}>"
            else:
                schema[c.txt] = f"<categorical: e.g. {vals[:5]}, ... ({len(vals)} known values)>"
    return schema


def build_opro_prompt(
    data,
    sorted_traj_desc,
    k,
    collisions,
    retry_error=None,
    prior_response=None,
):
    """Build the OPRO meta-prompt.

    `sorted_traj_desc` MUST be sorted DESCENDING by d2h (worst first, best LAST).
    DO NOT change to ascending — OPRO best-last is load-bearing: end-of-prompt
    examples receive the most attention weight, so the best-so-far rows must
    sit immediately before the "propose new candidates" instruction.
    """
    constraints_block = _format_constraints(data)
    y_desc = _format_objectives(data)
    traj_table = _format_trajectory(data, sorted_traj_desc)
    collisions_block = _format_collisions(data, collisions)
    best_so_far = sorted_traj_desc[-1][1] if sorted_traj_desc else float("inf")
    feat_schema = _format_feature_schema(data)

    schema_lines = [f'    "{name}": {spec}' for name, spec in feat_schema.items()]
    schema_str = "{\n" + ",\n".join(schema_lines) + "\n  }"

    system = (
        "You are an optimizer. Your job is to propose feature configurations "
        "that minimize d2h (Chebyshev distance to ideal objective values; lower is better). "
        "STRICT RULE: every value you propose MUST be inside the declared NUM range or "
        "the declared SYM allowed-values set. Do NOT use values from your prior knowledge "
        "of the domain — only the values listed in the constraints below are valid."
    )

    user_parts = [
        "## Hard constraints (each proposed value must satisfy these EXACTLY)",
        constraints_block,
        "",
        "## Objectives (NOT to be proposed; they are evaluated for you)",
        y_desc,
        "",
        "## Trajectory",
        "Configurations seen so far, sorted from worst to best (best is last):",
        traj_table,
        "",
    ]

    if collisions_block:
        user_parts += ["## Diversification", collisions_block]

    if retry_error is not None:
        user_parts += [
            "## Retry",
            f"Your previous response was not valid: {retry_error}",
            f"Prior response (do not repeat the same mistake):\n{prior_response}",
            "Re-read the hard constraints above. Every NUM must be in range; "
            "every SYM must be EXACTLY one of the listed values. No exceptions.",
            "",
        ]

    user_parts += [
        "## Reminder",
        "Before you respond, verify each value against the hard-constraints block. "
        "NUM values outside [lo, hi] and SYM values not in the allowed set will be "
        "rejected and waste a retry. Only propose values that are explicitly allowed.",
        "",
        "## Instruction",
        f"Propose exactly {k} new feature configurations whose d2h is below {best_so_far:.4f}.",
        "Respond with ONLY a JSON object of this exact shape (no prose, no markdown fences):",
        "{",
        '  "candidates": [',
        f"    {schema_str},",
        "    ...",
        "  ]",
        "}",
        f'The "candidates" array MUST have exactly {k} entries.',
    ]

    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def get_analysis_prompt(features_desc, best_rows, rest_rows):
    return [
        {
            "role": "system",
            "content": f"You are a data scientist. Analyze the patterns in this dataset.\nFeatures:\n{features_desc}",
        },
        {
            "role": "user",
            "content": f"""Analyze the differences between these Best examples:\n{best_rows}\n\nAnd these Rest examples:\n{rest_rows}\n\n
Output a structured analysis identifying:
1. Feature Ranking: List 3-5 most influential features.
2. Tradeoff Identification: Note any conflicting objectives.
3. Directionality: For continuous variables, which direction improves the objective?
""",
        },
    ]


def get_constraint_prompt(analysis):
    return [
        {
            "role": "user",
            "content": f"""Based on this analysis:\n{analysis}\n\n
Identify explicit boundaries for valid configurations. Distinguish between:
- Hard Constraints: Inviolable rules (physical/logical limits).
- Soft Constraints: Heuristic preferences.
List them clearly.
""",
        }
    ]


def get_generation_prompt(analysis, constraints, best_rows, features_desc, n_count=2):
    return [
        {
            "role": "user",
            "content": f"""
Analysis: {analysis}
Constraints: {constraints}
Configuration Features (generate values for ONLY these):
{features_desc}

Task: Generate {n_count} NEW better configurations and {n_count} NEW poorer configurations.
Strictly adhere to the Hard Constraints.
The "features" list in your output MUST include values for ONLY the configuration features listed above, in the same order. Do NOT include objective/output values.
Optimize the key features identified in the Analysis to improve the Objectives.

Return the output as a valid JSON object with keys "better_examples" and "poorer_examples", where each example has a "features" list.
Do not wrap in markdown blocks. Just return the raw JSON string.
Example format:
{{
  "better_examples": [{{"features": [0.1, 2, "A"]}}],
  "poorer_examples": [{{"features": [0.9, 5, "B"]}}]
}}
""",
        }
    ]


def get_validation_prompt(constraints, generated_examples):
    return [
        {
            "role": "user",
            "content": f"""
Constraints:
{constraints}

Generated Configuration:
{generated_examples}

Task: Verify if the configuration violates any Hard Constraints.
If VALID, output "VALID".
If INVALID, explain the violation and provide a REVISED configuration that satisfies the constraint while preserving other values.

Return JSON: {{"status": "VALID" | "INVALID", "revised_features": [...] (if invalid), "explanation": "..."}}
""",
        }
    ]


def bs_prompt(data, best, rest):
    feature_names = [col.txt for col in data.cols.x]

    def examples(rows):
        return "\n".join(
            "  - " + ", ".join(f"{k}={v}" for k, v in zip(feature_names, row)) for row in rows
        )

    best_examples_str = examples(best)
    rest_examples_str = examples(rest)
    return [
        (
            "system",
            "You are a helpful AI that generates synthetic data examples for optimization.",
        ),
        (
            "human",
            f"""Dataset features: {", ".join(feature_names)}

BEST examples (these have good performance):
{best_examples_str}

WORST examples (these have poor performance):
{rest_examples_str}

Generate 2 new examples that would be BETTER than the best examples, and 2 new examples that would be WORSE than the worst examples.

Return your answer as a simple list in this exact format:
BETTER:
- feature1=value1, feature2=value2, ...
- feature1=value1, feature2=value2, ...

WORSE:
- feature1=value1, feature2=value2, ...
- feature1=value1, feature2=value2, ...
        """,
        ),
    ]
