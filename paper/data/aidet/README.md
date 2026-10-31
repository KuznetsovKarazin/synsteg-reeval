# Trade-off and key-selection corpus (Sections 3.9-3.10 and 4.7-4.8 of the article)

`corpus.jsonl` was built by K. Chernov from the 299 GPT-4o records (`experiments/ai_detectors/build_corpus.py` in the method repository, no API calls): cover, human text (IMDB review truncated to the cover length), stego texts for b = 1..5, and 16 random key permutations at b = 5 (`cand_b5`). We rebuilt all non-human records bit for bit from the saved reports.

`scores/` holds the scores of detectors D1-D5 computed by K. Chernov (higher = more machine-like). Scripts: `paper/scripts/aidet_tradeoff.py`, `aidet_matrix.py`, `aidet_keysel_lr.py`, `fig_aidet.py`; outputs in `paper/results/aidet/`.
