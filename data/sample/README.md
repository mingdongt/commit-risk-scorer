# Sample scraped data

This directory holds a small, version-controlled **sample** of scraped GitHub PR
data so contributors and reviewers can see the data shape without re-running
the scraper.

## `example_prs.jsonl`

10 PRs scraped on 2026-05-10 with:

```bash
python -m src.data.scrape_github_prs \
    --repos psf/requests httpie/cli \
    --max-prs-per-repo 5 \
    --output data/sample/example_prs.jsonl
```

(no `GITHUB_TOKEN` — public unauthenticated rate limit was sufficient for this
sample size)

### Observed label distribution

| `ci_outcome` | Count |
| --- | --- |
| `passed` | 3 |
| `failed` | 3 |
| `mixed` | 1 |
| `unknown` | 3 |

The `unknown` label indicates PRs where neither check-runs nor a deterministic
status conclusion was available at scrape time — these are filtered out before
fine-tuning.

### Record schema (one per line, JSONL)

| Field | Type | Notes |
| --- | --- | --- |
| `pr_id` | string | `<owner>/<repo>#<number>` |
| `repo` | string | `<owner>/<repo>` |
| `pr_number` | int | |
| `title` | string | PR title |
| `base_sha`, `head_sha` | string | Commit SHAs |
| `merged` | bool | Whether the PR was merged |
| `ci_outcome` | string | `passed` / `failed` / `mixed` / `unknown` |
| `files_changed_count` | int | |
| `additions`, `deletions` | int | Line counts |
| `diff` | string | Unified diff text |

## Production scale

The labeled training set used for the production fine-tune target (Mistral-7B-v0.3
on NVIDIA NeMo + LoRA — see [`src/models/finetune/train_nemo.py`](../../src/models/finetune/train_nemo.py))
is a larger scrape of ~1k PRs across higher-volume repos (kubernetes/kubernetes,
django/django). That dataset lives in `data/raw/` and is gitignored — regenerate
with the command shown above using a `GITHUB_TOKEN`.
