# Direct AI generation of a seeded candidate dictionary

This bounded smoke test asked the model itself to produce a table from one fixed prompt and a public demo seed. It did not shuffle or select from an existing synonym dictionary.

Two fresh processes used `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, public seed `20261023`, CPU BF16, four compute threads and one interop thread. Sampling used temperature 0.8, top-p 0.95, top-k 50 and a 192-token ceiling. The exact prompt, rendered chat template, generation configuration, package versions and every input/output token ID are retained in the run records.

There were exactly two samples, run sequentially with identical settings. No prompt revision, retry, replacement or selection of a preferred output occurred. Both processes exited successfully and released their models.

| Observation | Run 1 | Run 2 |
|---|---:|---:|
| Generated tokens, including terminal token | 62 | 62 |
| Generation time | 10.815 s | 10.947 s |
| Whole-output JSON parsing | Pass | Pass |
| Exactly four groups of four lowercase ASCII words | Pass | Pass |
| Sixteen globally distinct words | Pass | Pass |

Input token IDs, generated token IDs, complete decoded text and ordered parsed JSON were identical. The two-process experiment took 42.258 seconds, including model loading and process setup.

Both unmodified outputs were:

```json
[
  ["good", "fine", "excellent", "super"],
  ["happy", "joyful", "elated", "cheerful"],
  ["fast", "swift", "quick", "nimble"],
  ["big", "large", "huge", "massive"]
]
```

This establishes an observed exact repeat for one prompt and one seed in this software and hardware environment. It does not establish reproducibility across platforms, dependency versions, models, prompts or other seeds. The checker validates JSON and lexical structure; it does not verify English vocabulary or semantic synonymy. A generated group is not a guarantee that its words can replace one another in arbitrary sentences.

The public seed is not a secret key. No payload was embedded, no attack was fitted or evaluated, and no secrecy, cryptographic security or communication performance claim follows. This experiment is separate from the frozen prospective v3 run and the cross-generator transfer pilot.

`run_smoke.py` is the complete runner. `launch.json` records the fixed design before sampling; `run_01.json` and `run_02.json` retain both full observations; `comparison.json` records exact comparisons and owned child exit codes. `environment_metadata.json` identifies the cached model/configuration and host resources. An independent read-only code review found no blocking issue in the structural checks or two-process execution.

The runner refuses to overwrite an existing launch or result. To perform an explicitly separate replication, copy the runner alone to a new directory and run it in the pinned neural environment with the existing model cache; that creates two new observations rather than replacing these records.
