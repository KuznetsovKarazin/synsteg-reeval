# Re-evaluation of LLM synonym-substitution steganography

Code and results of an independent re-evaluation of SynSteg, a linguistic steganography method in which a large language model writes a cover text and a table of synonyms, and this table is the secret key.

This repository accompanies the article "Is LLM Synonym-Substitution Steganography Undetectable? A Re-Evaluation of Steganalysis, Decoding and Key Cost" (O. Kuznetsov, K. Chernov, O. Smirnov, I. Ienina; submitted to *Computers*, MDPI). The original method is at https://github.com/po3na4skld/synsteg (commit f7709e6). This repository is a cleaned snapshot of the authors' working repository: working notes and drafts are not included.

## Layout

| Path | Contents |
|---|---|
| `src/synsteg_audit/` | Audit library: decoding and re-encoding checks, metrics, detector formulas, framed interval coding for the seed-key pilots |
| `scripts/` | Command-line entry points of the audit and of the pilots |
| `tests/` | Unit tests (`python -m pytest -q`) |
| `paper/data/reports/` | Saved SynSteg records of the four series (DS-2023, GPT-4o, luna, Qwen3): message, cover, stego text, key, decoded message. Copied unchanged from github.com/po3na4skld/synsteg, commit f7709e6 |
| `paper/data/artifacts/` | Sentence-level steganalysis corpora (DeepStego, 4331 pairs; GPT-4o, 1231 pairs) |
| `paper/scripts/`, `paper/results/` | Scripts and outputs of the article: protocol A/B summaries, replacement-rate sweep, interpretable detectors, key cost, ablation and transfer |
| `results/reproduction-2026-10-22-verified/` | Decoding audit of all 2175 saved records (article Table 8) |
| `results/neural-suite-2026-10-23/` | GPT-2 perplexity and detectors of machine-generated text (Table 10) |
| `results/framed-interval-*` | The four seed-key pilots (Table 11) |
| `experiments/`, `configs/`, `resources/`, `artifacts/` | Pre-registered protocols, configurations and resource manifests of the pilots |
| `data/source/` | The original SynSteg archive, stored in parts with SHA-256 checksums (see `ATTRIBUTION.md`) |

## Quick start

Python 3.12.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/import_source.py          # rebuilds and verifies the original archive
python -m pytest -q
python scripts/reproduce.py --output results/local-rerun
```

The article scripts in `paper/scripts/` read the saved records from `paper/data/reports` (or from `SYNSTEG_REPORTS`). They make no LLM calls.

```bash
pip install scikit-learn==1.9.1 wordfreq==3.1.1 matplotlib
python paper/scripts/summarize_protocols.py paper/results/steganalysis_sentences.json paper/results/steganalysis_missing.json
python paper/scripts/edits.py
python paper/scripts/sweep.py paper/results/sweep_results.json            # about 1.5 h on 4 cores
python paper/scripts/explain_and_key.py paper/results/explain_and_key_results.json
python paper/scripts/ablation_transfer.py paper/results/ablation_transfer_results.json
```

The neural detectors and the seed-key pilots need model weights (for example Qwen2.5-1.5B-Instruct in GGUF Q4_K_M and llama.cpp build b11791). The weights are not redistributed; the result folders record the exact files and their SHA-256 hashes.

## What is not included

No API keys or keys of live channels. Tests contain only offline fixtures.

## Licence

The original SynSteg code and the saved records in `paper/data/` keep their own licence (`LICENSE.upstream.txt`, non-commercial research use). The audit code, scripts and results added here are released under CC BY 4.0 unless stated otherwise in a file.

## Citation

See `CITATION.cff`. The archived release has a Zenodo DOI.
