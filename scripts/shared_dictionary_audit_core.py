#!/usr/bin/env python3
"""Independent shared-dictionary decoder and fixed-vector check.

This audit code never imports the experimental codec. Canonical resources,
token eligibility, normalization, HKDF and HMAC groups are reconstructed with
the standard library. Only the AES-SIV primitive uses cryptography.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import unicodedata

VERSION = 'synsteg-shared-dictionary-aes-siv-v1'
SCHEMA = 'synsteg-public-dictionary-v1'


def serialize(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(serialize(value)).hexdigest()


def dictionary_identity(data):
    if data.get('schema') != SCHEMA or not isinstance(data.get('version'), str) or not data['version']:
        raise ValueError('Invalid dictionary identity')
    classes, seen = {}, set()
    for row in data['classes']:
        name, words = row['id'], row['words']
        if not isinstance(name, str) or not name or name in classes or not isinstance(words, list) or len(words) < 2:
            raise ValueError('Invalid dictionary class')
        if any(not isinstance(word, str) or re.fullmatch('[a-z]{2,}', word) is None or word in seen for word in words):
            raise ValueError('Invalid or overlapping dictionary words')
        if len(set(words)) != len(words):
            raise ValueError('Repeated class member')
        seen.update(words)
        classes[name] = sorted(words)
    if not classes:
        raise ValueError('Empty dictionary')
    core = {'schema': SCHEMA, 'version': data['version'],
            'classes': [{'id': name, 'words': classes[name]} for name in sorted(classes)]}
    return digest(core), classes


def protocol_identity(protocol):
    if set(protocol) != {'version', 'min_group_size', 'max_bits_per_slot', 'framing'} or protocol['version'] != VERSION:
        raise ValueError('Invalid public protocol identity')
    if type(protocol['min_group_size']) is not int or protocol['min_group_size'] < 1:
        raise ValueError('Invalid minimum group size')
    if type(protocol['max_bits_per_slot']) is not int or protocol['max_bits_per_slot'] not in (1, 2):
        raise ValueError('Invalid slot width')
    if protocol['framing'] not in ('aes_siv', 'plain_length16'):
        raise ValueError('Invalid framing')
    return digest(protocol)


def analyze(text, dictionary, protocol):
    dictionary_sha, classes = dictionary_identity(dictionary)
    protocol_sha = protocol_identity(protocol)
    lookup = {word: name for name, words in classes.items() for word in words}
    markers, slots, cursor = [], [], 0
    for token_index, token in enumerate(re.finditer(r'\S+', text)):
        surface = token.group()
        letters = [index for index, char in enumerate(surface) if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']
        if not letters:
            continue
        start, end = letters[0], letters[-1] + 1
        word = surface[start:end]
        if re.fullmatch('[A-Za-z]+', word) is None:
            continue
        if any(char == '_' or unicodedata.category(char)[0] != 'P' for char in surface[:start] + surface[end:]):
            continue
        name = lookup.get(word.lower())
        if name is None:
            continue
        case = ('lower' if word == word.lower() else 'upper' if word == word.upper()
                else 'title' if word == word[0].upper() + word[1:].lower() else None)
        if case is None:
            continue
        width = 0
        for possible in range(1, protocol['max_bits_per_slot'] + 1):
            if len(classes[name]) >= protocol['min_group_size'] * 2 ** possible:
                width = possible
        if width == 0:
            continue
        absolute_start, absolute_end = token.start() + start, token.start() + end
        markers.extend([['literal', text[cursor:absolute_start]], ['class', name, case]])
        slots.append({'index': len(slots), 'token_index': token_index, 'start': absolute_start, 'end': absolute_end,
                      'surface': word, 'class_id': name, 'case_style': case, 'width': width})
        cursor = absolute_end
    markers.append(['literal', text[cursor:]])
    return {'dictionary_sha256': dictionary_sha, 'protocol_sha256': protocol_sha,
            'normalization_sha256': digest(markers), 'capacity_bits': sum(slot['width'] for slot in slots),
            'slots': slots, 'classes': classes}


def derive_keys(master, dictionary_sha, protocol_sha):
    if not isinstance(master, bytes) or len(master) != 32:
        raise ValueError('Exactly 32 key bytes required')
    salt = hashlib.sha256(b'synsteg/shared-dictionary/hkdf-salt/v1\x00' + bytes.fromhex(dictionary_sha)).digest()
    prk = hmac.new(salt, master, hashlib.sha256).digest()
    def expand(label, length):
        info = serialize([VERSION, protocol_sha, label])
        last, output, counter = b'', b'', 1
        while len(output) < length:
            last = hmac.new(prk, last + info + bytes([counter]), hashlib.sha256).digest()
            output += last
            counter += 1
        return output[:length]
    return expand('aead-aes-siv-512', 64), expand('mapping-hmac-sha256', 32)


def groups(analysis, slot, mapping_key):
    context = [VERSION, 'word-rank', analysis['dictionary_sha256'], analysis['protocol_sha256'],
               analysis['normalization_sha256'], slot['index'], slot['token_index'], slot['class_id'], slot['width']]
    ranked = sorted(analysis['classes'][slot['class_id']],
                    key=lambda word: (hmac.new(mapping_key, serialize(context + [word]), hashlib.sha256).digest(), word))
    return [ranked[label::2 ** slot['width']] for label in range(2 ** slot['width'])]


def aad(analysis, protocol, length):
    return [b'synsteg/shared-dictionary/authenticated-frame/v1', serialize({
        'version': VERSION, 'dictionary_sha256': analysis['dictionary_sha256'], 'protocol': protocol,
        'normalization_sha256': analysis['normalization_sha256'], 'payload_bytes': length})]


def decode(text, key, dictionary, protocol):
    analysis = analyze(text, dictionary, protocol)
    aead_key, mapping_key = derive_keys(key, analysis['dictionary_sha256'], analysis['protocol_sha256'])
    overhead = 144 if protocol['framing'] == 'aes_siv' else 16
    if analysis['capacity_bits'] < overhead:
        raise ValueError('Capacity below frame overhead')
    bits, needed, length, used = '', None, None, []
    for slot in analysis['slots']:
        membership = groups(analysis, slot, mapping_key)
        labels = [label for label, words in enumerate(membership) if slot['surface'].lower() in words]
        if len(labels) != 1:
            raise ValueError('Nonunique slot membership')
        bits += format(labels[0], '0' + str(slot['width']) + 'b')
        used.append({'slot': slot['index'], 'label': labels[0], 'candidates': membership[labels[0]]})
        if len(bits) >= 16 and needed is None:
            length = int(bits[:16], 2)
            needed = overhead + 8 * length
            if needed > analysis['capacity_bits']:
                raise ValueError('Header length exceeds capacity')
        if needed is not None and len(bits) >= needed:
            break
    if needed is None or len(bits) < needed or any(bit != '0' for bit in bits[needed:]):
        raise ValueError('Incomplete frame or nonzero padding')
    frame = int(bits[:needed], 2).to_bytes(needed // 8, 'big')
    if protocol['framing'] == 'aes_siv':
        from cryptography.hazmat.primitives.ciphers.aead import AESSIV
        payload = AESSIV(aead_key).decrypt(frame[2:], aad(analysis, protocol, length))
    else:
        payload = frame[2:]
    if len(payload) != length:
        raise ValueError('Decoded byte length mismatch')
    return {'payload_hex': payload.hex(), 'frame_hex': frame.hex(), 'frame_bits': needed,
            'padding_bits': len(bits) - needed, 'consumed_slots': len(used), 'analysis': analysis, 'used_groups': used}


def require(condition, name):
    if not condition:
        raise ValueError('Reference mismatch: ' + name)


def verify_reference(path):
    data = json.loads(Path(path).read_text())
    key = bytes.fromhex(data['key_hex'])
    checked = []
    for vector in data['vectors']:
        observed = decode(vector['stego_text'], key, data['dictionary'], vector['protocol'])
        analysis = observed['analysis']
        original = analyze(data['cover_text'], data['dictionary'], vector['protocol'])
        require(analysis['dictionary_sha256'] == data['dictionary_sha256'], 'dictionary_sha256')
        require(analysis['protocol_sha256'] == vector['protocol_sha256'], 'protocol_sha256')
        require(analysis['normalization_sha256'] == original['normalization_sha256'] == vector['normalization_sha256'], 'normalization_sha256')
        require(analysis['capacity_bits'] == original['capacity_bits'] == vector['capacity_bits'], 'capacity_bits')
        require(observed['payload_hex'] == data['payload_hex'], 'payload_hex')
        for field in ('frame_hex', 'frame_bits', 'padding_bits', 'consumed_slots'):
            require(observed[field] == vector[field], field)
        changed = 0
        for group in observed['used_groups']:
            index = group['slot']
            slot = analysis['slots'][index]
            candidates = [word.upper() if slot['case_style'] == 'upper' else word.capitalize()
                          if slot['case_style'] == 'title' else word for word in group['candidates']]
            require(slot['surface'] == min(candidates), 'canonical_choice')
            changed += slot['surface'] != original['slots'][index]['surface']
        require(changed == vector['changed_slots'], 'changed_slots')
        checked.append({'framing': vector['protocol']['framing'], 'independently_decoded': True,
                        'frame_bits': observed['frame_bits'], 'capacity_bits': analysis['capacity_bits']})
    return {'status': 'pass', 'reference_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'vectors': checked, 'experimental_codec_imported': False,
            'scope': 'Independent normalization, HKDF/HMAC mapping, framing and AES-SIV primitive decryption; no neural inference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vectors', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_reference(args.vectors), indent=2))


if __name__ == '__main__':
    main()
