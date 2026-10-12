# Working on this public code release

Preserve the public-data boundary: do not add course notes, assessment questions,
private benchmark packets, manuscripts, credentials, deployment hosts or model weights.
Use freshly authored generic examples for tests. Do not contact a model server,
Raspberry Pi, camera or cloud service during offline checks.

After code changes run `ruff check --fix .`, `ruff format .`, and `python -m pytest -q`.
Keep new Python functions annotated. Before committing, run the configured pre-commit
hooks. Preserve bundled third-party notices; do not assign a blanket software license.
