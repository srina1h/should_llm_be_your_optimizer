"""Synthetic-example prompt for the BS_LLM baseline and SynthCore.

`SYNTHETIC(data, best, rest).get_template_markdown()` is the live prompt: it
shows the model the best/rest split and asks for 2 better + 2 worse synthetic
configurations. The response is parsed back into rows by
`src.models.warm_start._markdown_to_rows`.
"""
from src.utils.ezr import rows, data, NUM, SYM


class SYNTHETIC():

    def __init__(self, i: data, best: rows, rest: rows):
        self.meta = {NUM: 'NUMBER', SYM: 'SYMBOL'}
        self.i = i
        self.dff = 0
        if len(i.cols.names) != len(i.cols.x) + len(i.cols.y):
            self.dff = len(i.cols.names) - len(i.cols.x) - len(i.cols.y)

        self.best = [b[:len(i.cols.x) + self.dff] for b in best]
        self.rest = [r[:len(i.cols.x) + self.dff] for r in rest]
        self.col_names = i.cols.names
        self.x = len(i.cols.x)

    def get_template_markdown(self):
        """Plain-text prompt (no JSON — robust for small models)."""
        feature_names = [col.txt for col in self.i.cols.x]

        best_examples_str = "\n".join(
            f"  - {', '.join(f'{name}={val}' for name, val in zip(feature_names, example))}"
            for example in self.best)
        rest_examples_str = "\n".join(
            f"  - {', '.join(f'{name}={val}' for name, val in zip(feature_names, example))}"
            for example in self.rest)

        messages = [
            ("system", "You are a helpful AI that generates synthetic data examples for optimization."),
            ("human", f'''Dataset features: {", ".join(feature_names)}

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
        '''),
        ]
        return messages
