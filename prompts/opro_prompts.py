"""Meta-prompt builder for OPRO (Yang et al., 2023) — LLM-as-Optimizer."""


# Cap vocab listings in the prompt for high-cardinality SYMs (typically numeric
# columns mis-classified as SYM by ezr when col name starts with a digit, e.g.
# `2NDFLRSF` or any SYM in a 17K-row dataset). Without this cap the constraints
# block + schema block can push prompts past gpt-oss-120b's 32K context window.
# Lenient parser already accepts out-of-vocab values, so truncation is safe —
# the snap step (dist=1.0 for SYM mismatch) remains correct.
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
        "Diversify — propose configurations distinct from these:\n"
        + "\n".join(lines)
        + "\n"
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
        '{',
        '  "candidates": [',
        f"    {schema_str},",
        "    ...",
        "  ]",
        '}',
        f'The "candidates" array MUST have exactly {k} entries.',
    ]

    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
