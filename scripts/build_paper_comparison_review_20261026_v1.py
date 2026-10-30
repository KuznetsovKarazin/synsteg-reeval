#!/usr/bin/env python3
"""Read-only synthesis of pinned reports and supplied manuscript/PDF.

No model, tokenizer, encoder, decoder, training or test data access.
This script requires the exact externally retrieved inputs, listed in its output.
It verifies identities and aggregates saved summary groups; it does not repeat
scientific evaluators. Outputs are new and must not already exist.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile

INPUTS = SOURCE_ROOT = OUT = IDENTITY_PATH = None
REPO = "KuznetsovKarazin/synsteg-independent-reproduction"
MAIN = "c4c2b7c0ba244f84632550e69457e0c8731d51df"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    sources = json.loads(IDENTITY_PATH.read_bytes())
    for s in sources:
        data = (INPUTS / s["repository_path"]).read_bytes()
        assert sha(data) == s["sha256"] and len(data) == s["bytes"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == s["git_blob_sha"]
        s.pop("ends_newline", None)
        s["url"] = f"https://github.com/{REPO}/blob/{s['commit']}/{s['repository_path']}"
        s["read_method"] = "fresh GitHub connector read at immutable commit; exact local UTF8 byte/hash verification"
    paper_names = {
        "02-article_uk.docx": "a34b5b567fd5650374d0264ac2db9a6787cfa370f7a2791d89399d864c61c178",
        "03-DeepStego-Privacy-Preserving-Natural-Language-Steganography-Using-Large-Language-Models-and-Advanced-Neural-Architectures.pdf": "4675b0d1e304a7257137f977d79e7a7e30b1efc4ceb78a8ee021f60ebc69c1ff",
    }
    papers = []
    for name, digest in paper_names.items():
        b = (SOURCE_ROOT / name).read_bytes()
        assert sha(b) == digest
        papers.append({"supplied_file": name, "sha256": digest, "bytes": len(b)})
    docx = SOURCE_ROOT / "02-article_uk.docx"
    with zipfile.ZipFile(docx) as z:
        tree = ET.fromstring(z.read("word/document.xml"))
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    m = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    paragraphs = ["".join(t.text or "" for t in p.iter() if t.tag in {f"{{{w}}}t", f"{{{m}}}t"})
                  for p in tree.findall(f".//{{{w}}}p")]
    assert len(paragraphs) == 506
    assert "L∈{128,256,512}" in paragraphs[55]
    assert "захищеним каналом" in paragraphs[19]
    assert "Qwen3-30B-A3B-Instruct-2507" in paragraphs[54]
    assert paragraphs[425] == "14.35×" and paragraphs[426] == "2.74×"
    summary = json.loads((INPUTS / "results/matched-archive-v1/summary.json").read_bytes())
    groups = [x for x in summary["groups"] if x["arm"] == "original_archived" and x["archived_payload_bits"] == "all"]
    assert {x["series"] for x in groups} == {"luna", "qwen"}
    assert sum(x["denominator"] for x in groups) == sum(x["full_exact"] for x in groups) == 375
    assert sum(x["requested_bits"] for x in groups) == sum(x["delivered_bits"] for x in groups) == 112000
    audit = json.loads((INPUTS / "results/matched-archive-v1-independent-audit/audit.json").read_bytes())
    assert audit["status"] == "passed" and audit["independent_decodes"]["original"] == 375
    model = json.loads((INPUTS / "results/model-status-inventory-20261026-v1/ledger.json").read_bytes())
    assert model["current_campaign"]["completed"] == 80
    claims = [
        {"id": "original_delivery", "author_locator": "DOCX §3.1, §4.1–4.2/Table2; paragraphs20,48,55–64,65–130",
         "author_reported": {"luna_final_exact": "300/300", "qwen_final_exact": "75/75", "payload_bits": [128,256,512], "receiver": "stego plus per-message synonym table transmitted via secure channel", "container_regeneration_allowed": True},
         "our_saved_result": {"accepted_archived_original_exact": "375/375", "requested_and_delivered_bits": 112000, "independent_original_decodes": 375},
         "comparison": "Agrees with deterministic recovery of the accepted archive; does not establish prospective first-attempt success or original model checkpoints.",
         "git_sources": ["results/matched-archive-v1/summary.json", "results/matched-archive-v1-independent-audit/audit.json"]},
        {"id": "author_first_container_accounting", "author_locator": "DOCX Table2; paragraphs80,88,96,104,112,120,130",
         "author_reported_first_second_third": {"luna": [[95,5,0],[97,3,0],[97,3,0]], "qwen": [[25,0,0],[22,2,1],[21,3,1]]},
         "derived_first_container_fit": {"luna": {"numerator": 289, "denominator": 300}, "qwen": {"numerator": 68, "denominator": 75}},
         "comparison": "Arithmetic of manuscript table only; fit of first container is not a complete first-attempt generation-and-key success endpoint."},
        {"id": "capacity", "author_locator": "DOCX Table2 paragraphs74–127; PDF p13 Table1",
         "author_reported_bpw": {"luna": [2.73,2.64,2.61], "qwen": [1.48,1.44,1.49], "gpt4o_cited_range": [4.09,4.34]},
         "our_saved_first80": {"public_primary_exact": "0/960", "public_secondary16_exact": "198/960", "distinct_covers_with_secondary_delivery": "66/80", "raw_v1_bits":3706,"raw_v2_bits":113,"v2_zero_covers":"36/80","v2_raw_range_bits":[0,12]},
         "comparison": "Different receiver resource and framing; 32-byte master key does not replace per-message codebook. Three payload assignments per cover are dependent. Raw bits are not delivered payload. Resource-blocked original/fixed arms are not measured failures.",
         "git_sources": ["results/prospective-development-transport-first80-v1-postrun/report_ru.md","docs/shared_dictionary_codec_v1_design.md"]},
        {"id": "ppl_selection", "author_locator": "DOCX §3.3 paragraphs50–52; §4.8/Table9 paragraphs414–471",
         "author_reported": {"gpt4o_pairs":300,"gpt2_median_stego_cover_ratio_base":14.35,"gpt2_median_stego_cover_ratio_selected":2.74,"payload_dependent_code_permutation":True},
         "our_saved_archive_measurement": {"base_gpt2_ppl_increased":"975/975","gpt4o_K32_improved_vs_base":"300/300","gpt4o_K32_worse_than_cover":"298/300","gpt4o_geometric_mean_base_cover":13.734,"gpt4o_geometric_mean_K32_cover":2.784435},
         "comparison": "Qualitative agreement: optimization partly repairs a large PPL increase. Manuscript medians and our geometric means are distinct aggregates; independent exact-number replication is not asserted. GPT2 is also optimizer, Qwen evaluates prefixes, and PPL is not semantic validity.",
         "git_sources": ["docs/results_neural_2026_10_23_ru.md"]},
        {"id": "detectability", "author_locator": "DOCX paragraphs60–62,412,471; PDF p13 Table2",
         "author_protocol": "SynSteg sentence-level random split; Conversation-Steganography whole-document split; paper itself calls this comparison indicative. D1–D5 are AI-versus-human against299 historical IMDB texts.",
         "our_saved_result": "Archived D1 often rises and D2–D5 often fall while PPL rises; these AI-origin scores do not establish cover-versus-stego security.",
         "comparison": "No common fresh source-grouped held-out detection endpoint on new covers has been established. AUC<0.5 is not evidence that stego is indistinguishable; 0.5 accuracy/0 recall is compatible with degenerate classification.",
         "git_sources": ["docs/results_neural_2026_10_23_ru.md","docs/comparison_scope_large_development_v3_ru.md"]},
        {"id": "new_models_and_candidate_builder", "author_locator": "DOCX paragraphs22–48,55",
         "author_models_as_declared": ["gpt-6-luna","Qwen3-30B-A3B-Instruct-2507 Q4_K_M via Ollama0.31.1"],
         "our_generation_snapshot": model["current_campaign"],
         "our_native_v3": {"fixed_chunks":2,"schema_accepted_chunks":1,"source_positions_in_accepted_chunk":5,"candidate_strings_in_accepted_chunk":40,"usable_candidate_rows":None,"capacity_bits":None},
         "comparison": "TinyLlama1.1B/Qwen0.5B covers and separate Qwen1.5B candidate pilots are different pipelines, not same-model replication. One valid JSON is not a usable dictionary or original/fixed comparison.",
         "git_sources": ["results/model-status-inventory-20261026-v1/ledger.json","docs/continuation_20261026_baseline80_transport80_mask_order_ru.md"]},
        {"id": "semantic_and_factuality", "author_locator": "DOCX paragraphs57–61; PDF p12 §4.1.2 and p14 Table3",
         "comparison": "Source faithfulness is support/contradiction relative to the exact supplied excerpt. An unsupported assertion may be true outside it; absolute truth requires separate evidence. Cover→stego semantic changes must be assessed separately from source→cover additions. PPL/readability/embedding score or exact decoding certifies neither.",
         "git_sources": ["docs/comparison_scope_large_development_v3_ru.md","docs/continuation_20261026_baseline80_transport80_mask_order_ru.md"]},
    ]
    issues = [
        {"id":"pdf_table1_enhanced_arithmetic", "source":"DeepStego PDF p13 Table1", "visually_verified":True,
         "printed_rows":[{"payload_bits":256,"words_required":52.00,"bpw":9.84},{"payload_bits":512,"words_required":29.55,"bpw":9.84}],
         "interpretation":"Displayed L/words does not match displayed9.84 in these rows. Do not silently repair/reassign table values; resolve against underlying raw records. This review does not infer the cause."},
        {"id":"pdf_mode_labels", "source":"DeepStego PDF p13 §4.2 and Table2", "visually_verified":True,
         "body_bits_per_word":[5,10],"table2_labels_bits_per_word":[1,4],"interpretation":"Protocol labels differ inside the published page; exact condition mapping unresolved."},
        {"id":"pdf_undetectability_claim", "source":"DeepStego PDF p13 Table2 and preceding paragraph", "visually_verified":True,
         "accuracy":0.5,"recall":0.0,"interpretation":"These two values do not prove complete undetectability and are compatible with a classifier always predicting cover on balanced data; actual predictions would be needed to diagnose it."},
        {"id":"archive_model_identity", "source":"DOCX paragraph55 versus audited model ledger", "interpretation":"Manuscript declares models; archived luna/Qwen records do not independently establish exact original checkpoints."},
    ]
    endpoint = {
        "scope":"Development diagnostic on fixed100 source groups; no closed test",
        "step1":"Finish fixed model×source generation ledger without replacement/selection; keep completed, failed, unknown, not_attempted distinct.",
        "step2":"Freeze a usable candidate-builder condition and pre-payload candidate pools for every generated cover; preserve syntax failures and zero eligible positions. If resources absent, original/fixed remains blocked, not0%failed.",
        "step3":"On identical covers and payload128/256/512bits, compare complete exact delivery and unconditional useful bits/word for original-compatible, fixed and public arms; report native framing and per-message/public resource bytes separately.",
        "step4":"Review every source→cover and successful cover→stego separately; predeclare labels and report all failures, not only attractive examples. Independent scored PPL is secondary and cannot stand in for factuality.",
        "step5":"For seeded generation require fresh receiver with only stego+key+public resources and invariant context/dictionary/eligibility after substitutions; two payload successes alone do not establish reliability/security.",
        "later":"Only a separately authorized held-out source-grouped cover-versus-stego evaluation can support generalized detectability claims; no such test is opened here."
    }
    ledger = {"schema":"paper-comparison-review-v1","review_id":"paper-comparison-review-20261026-v1","as_of_main":MAIN,
              "scope":"New read-only manuscript comparison; counts frozen to verified main snapshot, not live branch progress",
              "git_sources":sources,"supplied_primary_sources":papers,
              "docx_locator_definition":"P001..P506 enumerate all w:p descendants of word/document.xml in order, including table cells; concatenate w:t and m:t to preserve formulas. Stable paragraph locators, not rendered page numbers.",
              "pdf_locator_definition":"Printed page numbers out of19; p13 tables visually checked from supplied PDF rendered with Poppler.",
              "primary_pdf_doi":"https://doi.org/10.3390/computers14050165",
              "claims":claims,"unresolved_source_reporting_issues":issues,"next_minimal_comparison":endpoint,
              "new_model_calls":0,"new_tokenizer_calls":0,"new_encoder_decoder_calls":0,"paid_calls":0,"raw_closed_test_opened":False,
              "old_scientific_audits_reexecuted":False,"statistical_inference_performed":False,"git_mutations_by_reviewer":0}
    put("ledger.json",ledger)
    put("validation.json",{"schema":"paper-comparison-source-check-v1","status":"passed","git_sources_exact_blob_sha1_sha256_size_verified":len(sources),"supplied_source_hashes_verified":len(papers),"docx_paragraph_count":len(paragraphs),"math_text_preserved":True,"original_archive_summary_groups_checked":len(groups),"original_archive_exact":375,"original_archive_bits":112000,"saved_independent_decode_count_matches":True,"nn_calls":0,"scientific_audit_repeated":False,"limitations":"Source binding and saved-summary arithmetic only; not an independent re-execution of models/codec or a human-gold semantic review."})
    print(json.dumps({"status":"passed","git_sources":len(sources),"primary_sources":len(papers),"claims":len(claims),"unresolved_reporting_issues":len(issues)},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Fresh retrieved Git files, retaining repository-relative paths")
    parser.add_argument("--sources", type=Path, required=True,
                        help="Directory containing the supplied exact DOCX and PDF")
    parser.add_argument("--identities", type=Path, required=True,
                        help="Saved source-identities.json; hashes checked before reading claims")
    parser.add_argument("--output", type=Path, required=True,
                        help="New destination; existing output files are refused")
    args = parser.parse_args()
    INPUTS, SOURCE_ROOT, OUT, IDENTITY_PATH = args.inputs, args.sources, args.output, args.identities
    main()
