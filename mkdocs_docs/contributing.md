# Contributing

---

## Setup

```bash
git clone https://github.com/your-org/palingenesis.git && cd palingenesis
uv pip install -e ".[all]" --group dev
```

## Tests

```bash
pytest tests/               # all 87 tests
pytest tests/ -x            # stop on first failure
pytest tests/test_integration.py -v  # verbose single file
```

## Format

```bash
black --line-length 120 src/ tests/
```

## Docs

```bash
pip install mkdocs-material
mkdocs serve                # → http://localhost:8000
```

## Pull requests

1. Fork, branch from `main`
2. Write tests for new features (see `tests/` for patterns)
3. Run `pytest` and `black` before pushing
4. Keep PRs focused — one feature or fix per PR
5. Reference the relevant paper (arxiv ID) in code comments

## Architecture rules

- No circular imports. Dependency flows downward (see [architecture overview](architecture/overview.md)).
- Every optimization cites its paper in a comment block.
- Absolute imports only (`from palingenesis.config import Config`, never `from .config`).
- New config fields go in the typed dataclass in `config.py` with a comment explaining the default.
