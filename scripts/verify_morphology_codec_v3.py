#!/usr/bin/env python3
"""Real new-dictionary synthetic roundtrips; no natural-text quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as codec
from audit_universal_context_development_v1 import isolated_receive


def run(root):
    dictionary = base.PublicDictionary.from_dict(json.loads((root / 'data/shared_dictionary_morphology_v3.json').read_bytes()))
    policy = codec.UniversalPolicy.from_dictionaries(dictionary, dictionary,
        version='morphology-complete-tag-spectrum-v3-reference-v1')
    words = [c.words[0] for c in dictionary.classes]
    cover = 'Calibration: an ' + words[0] + '.\n' + ' '.join(words * 2)
    key = hashlib.sha256(b'PUBLIC-MORPHOLOGY-V3-SYNTHETIC-KEY').digest()
    records = []
    for framing, payload in [('plain_length16', b'\x01'), ('plain_length16', b'\x01\x02'), ('aes_siv', b'\x01')]:
        protocol = codec.UniversalProtocol(framing=framing)
        encoded = codec.encode(cover, payload, key, dictionary, policy, protocol)
        decoded, receiver = isolated_receive(root, encoded.stego_text, key, dictionary, policy, framing)
        assert decoded == payload
        assert codec.analyze_text(cover, dictionary, policy, protocol) == codec.analyze_text(encoded.stego_text, dictionary, policy, protocol)
        records.append({'framing': framing, 'payload_hex': payload.hex(), 'decoded_hex': decoded.hex(),
            'capacity_bits': encoded.capacity_bits, 'changed_slots': encoded.changed_slots,
            'stego_sha256': hashlib.sha256(encoded.stego_text.encode()).hexdigest(), 'receiver': receiver})
    return {'schema': 'morphology-v3-synthetic-codec-validation-v1', 'dictionary_sha256': dictionary.sha256,
        'policy_sha256': policy.sha256, 'cover_sha256': hashlib.sha256(cover.encode()).hexdigest(),
        'synthetic_word_list_not_natural_cover': True, 'semantic_quality_assessed': False,
        'new_generator_calls': 0, 'arms': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.root.resolve())
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True, indent=2))
