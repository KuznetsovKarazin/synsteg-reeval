#!/usr/bin/env python3
"""Build a NEW strict WordNet baseline with monosemy and morphology screening.

Standard-library-only, fixed WordNet 3.0 bytes and source-only selection rules.
Global WordNet monosemy does NOT imply contextual substitutability. The derived
JSON is compatible with PublicDictionary and embeds the upstream license. No
experimental covers, LLMs, or paid services are used. V1 remains unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "wordnet-recovery-v2"
SCHEMA = "synsteg-public-dictionary-v1"
SOURCE_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
SOURCE_SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
SOURCE_BYTES = 10_775_600
POS_NAMES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}
SENSE_POS = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "a"}
SENSE_TYPE = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "s"}
# One-step WordNet morphological substitutions, with candidate bases checked
# against the source POS lexicon. We apply all POS rules, not a sender-side tag.
# No morphological variants are added to the shared dictionary.
MORPH_RULES = {
    "n": (("s", ""), ("ses", "s"), ("ves", "f"), ("xes", "x"),
          ("zes", "z"), ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y")),
    "v": (("s", ""), ("ies", "y"), ("es", "e"), ("es", ""),
          ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", "")),
    "a": (("er", ""), ("est", ""), ("er", "e"), ("est", "e")),
    "r": (),
}
WORD = re.compile(r"[a-z]{2,}\Z", re.ASCII)
SENSE_KEY = re.compile(r"(.+)%([1-5]):([0-9]{2}):([0-9]{2}):([^:]*):([0-9]*)\Z", re.ASCII)
# Curated grammatical/function-word exclusions, not a frequency stopword list.
# This is intentionally explicit and frozen, with no external stopword package.
DENYLIST = frozenset("""
about above across after against all along although am among an and any are
around as at be because been before being below beneath beside besides between
beyond both but by can cannot could dare despite did do does doing done down
during each either else enough even ever every everybody everyone everything
except few for from had has have having he hence her here hers herself him
himself his how however if in inside into is it its itself less lest many may
me might mine more most much must my myself near need neither never no nobody
none nor not nothing of off on once one oneself only onto or other others
otherwise ought our ours ourselves out outside over own per quite rather round
same several shall she should since so some somebody someone something such
than that the their theirs them themselves then thence there therefore these
they this those though through throughout thus till to together too toward
towards under underneath unless unlike until unto up upon us used very via was
we were what whatever when whence whenever where whereas wherever whether
which whichever while who whoever whom whose why will with within without
would yet you your yours yourself yourselves
""".split())
REQUIRED_MEMBERS = (
    "wordnet/LICENSE", "wordnet/index.sense",
    *(f"wordnet/data.{name}" for name in POS_NAMES.values()),
    *(f"wordnet/{name}.exc" for name in POS_NAMES.values()),
)


class BuildError(ValueError):
    """Invalid or untrusted source, or a refusal to replace existing outputs."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def builder_source_sha256() -> str:
    """Hash source with LF line endings so Windows Git settings do not alter it."""
    return sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))


def json_bytes(value: object) -> bytes:
    """Stable human-readable UTF-8/ASCII JSON with LF on every operating system."""
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("ascii")


def rules() -> dict:
    return {
        "version": VERSION,
        "lemma_form": "unmodified data-file lemma matching lowercase ASCII [a-z]{2,}",
        "adjective_position_markers": "reject marked lemmas (including (a), (p), (ip))",
        "sense_rule": "exactly one POS and exactly one (POS,synset offset) across ALL index.sense entries for the lowercase spelling, including untagged senses; satellites share adjective POS; selected entry also has sense_number == 1",
        "frequency_rule": "index.sense tag_cnt >= 1; historical semantic-concordance counts, not modern frequency",
        "missing_sense_index": "reject data-file lemma variants without an exact POS/offset/lemma/lexical-ID sense-index entry",
        "candidate_class": "eligible lemmas from exactly one WordNet 3.0 synset; at least two",
        "global_disjointness": "global unique POS/synset membership makes overlap impossible; assert, never greedily assign ambiguous spellings",
        "morphology_rule": "reject spelling with any distinct exception base in ANY source POS, or any distinct one-step suffix-derived base present in that rule's POS index.sense lexicon; no recursive stemming, word generation, contextual POS tagger, or cover input",
        "morphology_substitutions": {pos: [list(pair) for pair in pairs] for pos, pairs in MORPH_RULES.items()},
        "morphology_exception_policy": "all distinct bases in noun/verb/adj/adv.exc count, including source exceptions whose base lacks an index.sense entry; self-maps do not count",
        "morphology_interpretation": "no alternative WordNet base detected, not a proof of matching grammatical number/tense or absence of lexicalized inflection",
        "serialization": "class IDs and words lexicographically sorted; ensure_ascii JSON, sorted keys, indent 2, final LF",
        "class_id": "wn30-{POS n/v/a/r}-{8-digit synset offset}",
        "denylist": sorted(DENYLIST),
        "ai_review": False,
        "experimental_cover_input": False,
        "development_feedback": "motivates this separate stricter version; no per-observed-word blacklist or output-dependent choice",
    }


def load_source(path: Path) -> dict[str, bytes]:
    if not path.is_file():
        raise BuildError(f"WordNet archive not found: {path}. Use --download or --wordnet-zip PATH.")
    raw = path.read_bytes()
    actual = sha256(raw)
    if len(raw) != SOURCE_BYTES or actual != SOURCE_SHA256:
        raise BuildError(f"WordNet archive identity mismatch: {len(raw)} bytes, sha256={actual}; expected {SOURCE_BYTES} bytes, sha256={SOURCE_SHA256}. No output written.")
    try:
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in REQUIRED_MEMBERS}
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise BuildError(f"Cannot read required WordNet members: {exc}") from exc


def download_source(path: Path) -> None:
    """Download only with explicit CLI --download and verify before saving."""
    if path.exists():
        load_source(path)
        return
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read(SOURCE_BYTES + 1)
    except OSError as exc:
        raise BuildError(f"WordNet download unavailable: {exc}. Supply the pinned archive locally with --wordnet-zip.") from exc
    if len(raw) != SOURCE_BYTES or sha256(raw) != SOURCE_SHA256:
        raise BuildError("Downloaded WordNet archive failed fixed size/SHA256 checks; nothing saved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects an archive supplied during the download.
    with path.open("xb") as handle:
        handle.write(raw)


def _decode(members: dict[str, bytes], name: str) -> str:
    try:
        return members[name].decode("ascii")
    except (KeyError, UnicodeError) as exc:
        raise BuildError(f"Missing or non-ASCII source member: {name}") from exc


def parse_senses(text: str) -> dict[tuple[str, int, str, int], dict]:
    """Parse index.sense without rebuilding satellite sense keys or lemmatizing."""
    result = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        try:
            key, offset_text, number_text, count_text = line.split()
            match = SENSE_KEY.fullmatch(key)
            if match is None or not re.fullmatch(r"[0-9]{8}", offset_text):
                raise ValueError("bad sense key/offset")
            word, type_id, lexfile, lexid, _head, _headid = match.groups()
            offset, number, count = int(offset_text), int(number_text), int(count_text)
            if number < 1 or count < 0:
                raise ValueError("negative count or invalid sense number")
            identity = (SENSE_POS[type_id], offset, word, int(lexid))
            if identity in result:
                raise ValueError("duplicate sense identity")
            result[identity] = {"sense_key": key, "sense_number": number, "tag_count": count, "lexfile": int(lexfile), "synset_type": SENSE_TYPE[type_id]}
        except (ValueError, KeyError) as exc:
            raise BuildError(f"Malformed index.sense line {lineno}: {exc}") from exc
    return result


def source_lexicon(senses: dict) -> tuple[dict[str, set], dict[str, set]]:
    memberships = defaultdict(set)
    lexicon = {pos: set() for pos in POS_NAMES}
    for pos, offset, word, _lexid in senses:
        memberships[word].add((pos, offset))
        lexicon[pos].add(word)
    return dict(memberships), lexicon


def parse_exceptions(members: dict[str, bytes]) -> dict[str, dict[str, tuple[str, ...]]]:
    exceptions = {}
    for pos, name in POS_NAMES.items():
        entries = defaultdict(set)
        for lineno, line in enumerate(_decode(members, f"wordnet/{name}.exc").splitlines(), 1):
            parts = line.split()
            if len(parts) < 2:
                raise BuildError(f"Malformed {name}.exc line {lineno}: expected form and base")
            entries[parts[0]].update(parts[1:])
        exceptions[pos] = {word: tuple(sorted(bases)) for word, bases in entries.items()}
    return exceptions


def morphology_evidence(word: str, lexicon: dict[str, set], exceptions: dict) -> list[dict]:
    """A source-only overinclusive exclusion, not a context-dependent lemmatizer."""
    found = {}
    for pos in POS_NAMES:
        for base in exceptions[pos].get(word, ()):
            if base != word:
                found[(pos, base, "exception", "")] = {"pos": pos, "base": base, "kind": "exception"}
        for suffix, replacement in MORPH_RULES[pos]:
            if word.endswith(suffix):
                base = word[:-len(suffix)] + replacement
                if base != word and base in lexicon[pos]:
                    rule = f"{suffix}->{replacement}"
                    found[(pos, base, "suffix", rule)] = {"pos": pos, "base": base, "kind": "suffix", "rule": rule}
    return [found[key] for key in sorted(found)]


def static_exclusions(word: str, memberships: dict, lexicon: dict, exceptions: dict) -> dict:
    senses = sorted(memberships.get(word, ()))
    reasons = []
    if len({pos for pos, _offset in senses}) != 1:
        reasons.append("not_exactly_one_pos")
    if len(senses) != 1:
        reasons.append("not_exactly_one_synset")
    morphology = morphology_evidence(word, lexicon, exceptions)
    if morphology:
        reasons.append("potential_inflection")
    return {"reasons": reasons, "memberships": [[pos, f"{offset:08d}"] for pos, offset in senses],
            "morphology_evidence": morphology}


def build(members: dict[str, bytes]) -> tuple[dict, dict]:
    """Pure deterministic transformation. CLI verifies archive bytes beforehand."""
    license_text = _decode(members, "wordnet/LICENSE")
    if "WordNet Release 3.0" not in license_text or "Princeton University" not in license_text:
        raise BuildError("Missing WordNet 3.0 license notice")
    senses = parse_senses(_decode(members, "wordnet/index.sense"))
    memberships, lexicon = source_lexicon(senses)
    exceptions = parse_exceptions(members)
    exclusions_cache = {}
    counts = Counter()
    candidates = []
    seen_synsets = set()
    for pos, name in POS_NAMES.items():
        for lineno, line in enumerate(_decode(members, f"wordnet/data.{name}").splitlines(), 1):
            if not line or line.startswith("  "):
                continue
            try:
                lexical, sep, gloss = line.partition("|")
                parts = lexical.split()
                offset, lexfile, ss_type, word_count = int(parts[0]), int(parts[1]), parts[2], int(parts[3], 16)
                if not sep or word_count < 1 or len(parts) < 5 + 2 * word_count:
                    raise ValueError("truncated synset record")
                allowed_types = ("a", "s") if pos == "a" else (pos,)
                if ss_type not in allowed_types:
                    raise ValueError("part-of-speech mismatch")
                identity = (pos, offset)
                if identity in seen_synsets:
                    raise ValueError("duplicate synset identity")
                seen_synsets.add(identity)
                eligible = []
                for index in range(word_count):
                    raw_word, lex_id_text = parts[4 + index * 2:6 + index * 2]
                    lex_id = int(lex_id_text, 16)
                    counts["lemmas_seen"] += 1
                    if WORD.fullmatch(raw_word) is None:
                        counts["rejected_form"] += 1
                        continue
                    if raw_word in DENYLIST:
                        counts["rejected_denylist"] += 1
                        continue
                    sense = senses.get((pos, offset, raw_word, lex_id))
                    if sense is None:
                        counts["rejected_missing_exact_sense_index"] += 1
                        continue
                    if sense["lexfile"] != lexfile or sense["synset_type"] != ss_type:
                        raise ValueError(f"inconsistent sense-index entry for {raw_word}")
                    if sense["sense_number"] != 1:
                        counts["rejected_nonfirst_sense"] += 1
                        continue
                    if sense["tag_count"] < 1:
                        counts["rejected_zero_tag_count"] += 1
                        continue
                    counts["eligible_before_strict_filters"] += 1
                    exclusion = exclusions_cache.setdefault(raw_word, static_exclusions(raw_word, memberships, lexicon, exceptions))
                    if exclusion["reasons"]:
                        # Reasons can overlap; these counters are not a partition.
                        for reason in exclusion["reasons"]:
                            counts[f"strict_reason_{reason}"] += 1
                        counts["rejected_strict_lemma_occurrences"] += 1
                        continue
                    eligible.append({"word": raw_word, "sense_key": sense["sense_key"], "sense_number": 1, "tag_count": sense["tag_count"]})
                    counts["eligible_lemma_senses"] += 1
                if len({item["word"] for item in eligible}) != len(eligible):
                    raise ValueError("duplicate eligible lemma within synset")
                if len(eligible) >= 2:
                    candidates.append({"id": f"wn30-{pos}-{offset:08d}", "pos": pos, "offset": offset, "synset_type": ss_type, "gloss": gloss.strip(), "members": sorted(eligible, key=lambda item: item["word"])})
            except (ValueError, IndexError) as exc:
                raise BuildError(f"Malformed data.{name} line {lineno}: {exc}") from exc
    counts["synsets_seen"] = len(seen_synsets)
    counts["candidate_classes"] = len(candidates)
    used = set()
    selected = []
    for candidate in candidates:
        words = {member["word"] for member in candidate["members"]}
        if words & used:
            raise BuildError("Strict unique-membership rules unexpectedly produced overlapping classes")
        selected.append(candidate)
        used.update(words)
    selected.sort(key=lambda item: item["id"])
    if not selected:
        raise BuildError("No classes survive the frozen selection rules")
    classes = [{"id": item["id"], "words": [word["word"] for word in item["members"]]} for item in selected]
    core = {"schema": SCHEMA, "version": VERSION, "classes": classes}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    dictionary = {**core, "provenance": {"source": "Princeton WordNet 3.0, NLTK wordnet.zip distribution", "source_url": SOURCE_URL, "source_archive_sha256": SOURCE_SHA256, "rules_sha256": sha256(json_bytes(rules())), "recovery_status": "new strict source-only baseline; not recovery v1 or the lost shared_dictionary_ai_v2.json; no AI review", "semantic_scope": "WordNet monosemy and no detected alternate inflection base do not guarantee contextual substitutability"}, "license": {"name": "Princeton WordNet 3.0 license", "text": license_text}}
    counts["classes"] = len(classes)
    counts["words"] = len(used)
    provenance = {
        "schema": "synsteg-dictionary-recovery-provenance-v2",
        "version": VERSION,
        "rules": rules(),
        "rules_sha256": sha256(json_bytes(rules())),
        "source": {"url": SOURCE_URL, "archive_sha256": SOURCE_SHA256, "archive_bytes": SOURCE_BYTES, "members": {name: {"sha256": sha256(members[name]), "bytes": len(members[name])} for name in sorted(REQUIRED_MEMBERS)}},
        "dictionary": {"canonical_sha256": sha256(canonical), "file_sha256": sha256(json_bytes(dictionary)), "file_bytes": len(json_bytes(dictionary))},
        "counts": dict(sorted(counts.items())),
        "classes_by_pos": dict(sorted(Counter(item["pos"] for item in selected).items())),
        "class_size_histogram": dict(sorted(Counter(str(len(item["words"])) for item in classes).items(), key=lambda item: int(item[0]))),
        "selected_classes": selected,
        "exclusion_reason_counts_are_nonexclusive": True,
        "source_lexicon_spellings": len(memberships),
        "wordnet_license_sha256": sha256(members["wordnet/LICENSE"]),
    }
    return dictionary, provenance


def write_outputs(dictionary: dict, provenance: dict, output: Path, provenance_dir: Path) -> None:
    if output.name in {"shared_dictionary_ai_v1.json", "shared_dictionary_ai_v2.json", "shared_dictionary_v1.json", "shared_dictionary_recovery_v1.json"}:
        raise BuildError("Refusing to reuse a historical dictionary filename for this new version")
    provenance = {**provenance, "builder_sha256": builder_source_sha256(), "builder_hash_format": "source bytes with CRLF normalized to LF"}
    outputs = [
        (output, json_bytes(dictionary)),
        (provenance_dir / "provenance.json", json_bytes(provenance)),
        (provenance_dir / "WORDNET-LICENSE.txt", dictionary["license"]["text"].encode("ascii")),
    ]
    normalized = [path.resolve() for path, _raw in outputs]
    if len(set(normalized)) != len(normalized):
        raise BuildError("Output paths overlap")
    # Check all existing files before the first write; identical reruns are safe.
    for path, raw in outputs:
        if path.exists() and (not path.is_file() or path.read_bytes() != raw):
            raise BuildError(f"Refusing to overwrite differing existing output: {path}; choose fresh output paths")
    for path, raw in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordnet-zip", type=Path, default=ROOT / ".cache/wordnet-recovery-v2/wordnet.zip")
    parser.add_argument("--download", action="store_true", help="download the pinned public WordNet archive if absent")
    parser.add_argument("--output", type=Path, default=ROOT / "data/shared_dictionary_recovery_v2.json")
    parser.add_argument("--provenance-dir", type=Path, default=ROOT / "results/dictionary-recovery-v2")
    args = parser.parse_args(argv)
    try:
        if args.download:
            download_source(args.wordnet_zip)
        dictionary, provenance = build(load_source(args.wordnet_zip))
        write_outputs(dictionary, provenance, args.output, args.provenance_dir)
    except (BuildError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "built", "version": VERSION, "output": str(args.output), "provenance_dir": str(args.provenance_dir), **provenance["dictionary"], "classes": provenance["counts"]["classes"], "words": provenance["counts"]["words"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
