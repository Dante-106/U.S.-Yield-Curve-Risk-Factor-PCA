# Contributing

Changes must preserve the explicit units, sign conventions, time ordering, data provenance, and intended-use boundaries documented in the model card.

Before opening a review:

1. add or update tests for normal, boundary, and failure paths;
2. run `ruff check src tests scripts` and `pytest -q`;
3. rebuild and execute the notebook top-to-bottom;
4. update the model card, validation report, runbook, and changelog when behavior or use changes;
5. include numerical before/after evidence and state whether golden results intentionally changed;
6. identify model-risk impact: data, methodology, implementation, use, monitoring, or governance.

Executable Python and notebook code cells must contain no comment tokens. Express implementation intent through precise names, small functions, type contracts, docstrings, tests, and durable model documentation. The tokenizer-based release test enforces this rule without removing API documentation or governance disclosures.

Never weaken or silence a control merely to make a run pass. Threshold changes require evidence, documented impact, and independent review. Do not commit credentials, proprietary positions, unapproved market data, generated caches, or run outputs.
