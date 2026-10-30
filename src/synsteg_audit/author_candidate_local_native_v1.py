"""Two-request Qwen candidate-provider diagnostic, separate from author API.

No native process or tokenization at import. Existing EOS-masked top50 bridge
is used deliberately: stop is first complete raw JSON value or512 committed
tokens, never EOS. Every physical attempt is cold, bounded and journaled.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import time

from .author_candidate_local_adapter_v1 import canonical, digest, parse_response
from .framed_interval_llm_v1_journal import atomic_json, fsync_directory
from .framed_interval_public_v1_native import (
    PublicOwnedBridge, verify_files, verify_resident, public_digest,
    RESOURCE_ROOT, BRIDGE, MODEL, CPU, validate_profile,
)
from .seeded_pair_llm import quantize

VERSION = 'author-candidate-local-native-v1'
CAP = 512
ATTEMPT_SECONDS = 600
TOTAL = 1 << 24
EXPECTED_REQUESTS = [
 '2a510a01e1b7ee830f393c97bce447889f28ff24e251f9c6ee82231259bc2247',
 'ec61d91995330841f842969e930c996004214e8822b6511c8f81383ae860d328',
]
PROTOCOL = 'experiments/author-candidate-local-native-v1/protocol.json'
REQUESTS = 'experiments/author-candidate-local-native-v1/requests.json'
PROFILE = 'experiments/framed-interval-public-v1/native-public-profile.json'
CLOSURE = 'experiments/author-candidate-local-native-v1/source-closure.json'
INTENT = 'experiments/author-candidate-local-native-v1/execution-intent-v1.json'
REQUIRED_CLOSURE = frozenset(['experiments/author-candidate-local-native-v1/execution-intent-v1.json', 'experiments/author-candidate-local-native-v1/protocol.json', 'experiments/author-candidate-local-native-v1/requests.json', 'experiments/author-candidate-local-native-v1/run_native.py', 'experiments/framed-interval-public-v1/native-public-profile.json', 'scripts/prepare_author_candidate_local_native_v1.py', 'src/synsteg_audit/__init__.py', 'src/synsteg_audit/author_candidate_local_adapter_v1.py', 'src/synsteg_audit/author_candidate_local_native_v1.py', 'src/synsteg_audit/canonical_channel_v2.py', 'src/synsteg_audit/framed_interval_llm_v1.py', 'src/synsteg_audit/framed_interval_llm_v1_journal.py', 'src/synsteg_audit/framed_interval_llm_v1_qwen.py', 'src/synsteg_audit/framed_interval_public_v1.py', 'src/synsteg_audit/framed_interval_public_v1_native.py', 'src/synsteg_audit/integer_interval_channel_v1.py', 'src/synsteg_audit/proc_namespace_identity_v1.py', 'src/synsteg_audit/proc_namespace_identity_v2.py', 'src/synsteg_audit/seeded_pair_llm.py'])


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def write_bytes(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(Path(path).parent)


class Journal:
    def __init__(self, path):
        self.stream = Path(path).open('xb')
        fsync_directory(Path(path).parent)
        self.previous = '0' * 64
        self.sequence = 0

    def append(self, event):
        body = {'sequence': self.sequence, 'previous_sha256': self.previous, **event}
        row = {**body, 'event_sha256': digest(canonical(body))}
        self.stream.write(canonical(row) + b'\n')
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.previous = row['event_sha256']; self.sequence += 1
        return row['event_sha256']

    def close(self):
        self.stream.close()


def chatml(request):
    return ('<|im_start|>system\n' + request['system'] + '<|im_end|>\n'
            '<|im_start|>user\n' + request['user'] + '<|im_end|>\n'
            '<|im_start|>assistant\n')


def validate_request(request, chunk_order):
    if type(chunk_order) is not int or chunk_order not in (0, 1):
        raise ValueError('only fixed chunk0/1')
    fields = {'chunk_order','word_start','words','system','user','temperature_argument',
              'json_shape_attempt_limit','request_sha256'}
    if type(request) is not dict or set(request) != fields:
        raise ValueError('closed request schema')
    body = {k:v for k,v in request.items() if k != 'request_sha256'}
    if (request['chunk_order'] != chunk_order or request['request_sha256'] != EXPECTED_REQUESTS[chunk_order]
            or digest(canonical(body)) != request['request_sha256']):
        raise ValueError('fixed request identity differs')


def seed_hex(request_sha256, attempt_order):
    if type(attempt_order) is not int or not 0 <= attempt_order < 3:
        raise ValueError('attempt order must be0..2')
    return digest(canonical([VERSION,'fixed-public-diagnostic-seed',request_sha256,attempt_order]))


def choose(weights, seed, request_sha256, attempt_order, ids):
    if sum(x['weight'] for x in weights) != TOTAL:
        raise ValueError('categorical total differs')
    context = [VERSION,request_sha256,attempt_order,len(ids),digest(canonical(ids))]
    raw = hmac.new(bytes.fromhex(seed), canonical(context), hashlib.sha256).digest()
    value = int.from_bytes(raw[:3], 'big')  # exactly uniform on2^24 under PRF assumption
    remaining = value
    for row in weights:
        if remaining < row['weight']:
            return row['token_id'], value, digest(canonical(context))
        remaining -= row['weight']
    raise ValueError('inconsistent categorical law')


def completion(raw):
    """Whole raw prefix only; no fence removal, scanning, repair or re-rendering."""
    try: text = raw.decode('utf-8')
    except UnicodeDecodeError: return False, None
    try: json.loads(text)
    except RecursionError: return True, 'parser_resource_refusal'
    except (ValueError, TypeError): return False, None
    return True, 'complete_json_value'


def production_bridge(root, profile, directory, deadline):
    resource_check = verify_files(root, profile)
    atomic_json(directory/'resource-check.json', resource_check)
    base = Path(root)/RESOURCE_ROOT
    bridge = PublicOwnedBridge(base/BRIDGE,base/MODEL,base/CPU,directory/'native.stderr.log',deadline,
        expected_sha256=profile['resource_files'][BRIDGE]['sha256'],profile_sha256=public_digest(profile))
    try:
        ready = {'vocabulary_size':bridge.vocabulary_size,'masked_tokens':bridge.masked_tokens,
                 'context_tokens':bridge.context_tokens}
        if ready != profile['ready']: raise ValueError('actual READY differs')
        resident = verify_resident(root,profile,bridge)
        atomic_json(directory/'native-ready.json',{'ready':ready,'resident':resident,
            'resource_profile_sha256':public_digest(profile),'prompt_calls':0,'step_calls':0})
        return bridge
    except BaseException:
        bridge.close()
        raise


def run_attempt(root, profile, request, attempt_order, directory, *, bridge_factory=production_bridge):
    """One physical request, no retries. All error/partial evidence remains."""
    validate_request(request,request['chunk_order'])
    seed = seed_hex(request['request_sha256'],attempt_order)
    directory=Path(directory);directory.mkdir(exist_ok=False);fsync_directory(directory.parent)
    launch={'schema':VERSION+'-attempt-launch','chunk_order':request['chunk_order'],
        'attempt_order':attempt_order,'request_sha256':request['request_sha256'],
        'seed_hex':seed,'token_cap':CAP,'deadline_seconds':ATTEMPT_SECONDS,
        'prompt_text_sha256':digest(chatml(request).encode()),'request':request}
    atomic_json(directory/'launch.json',launch)
    journal=Journal(directory/'journal.jsonl')
    bridge=None;ids=[];raw=b'';prompt_ids=[];outcome='unknown';reason=None;accepted=None
    error=None;pending_step=False;prompt_invoked=False;prompt_committed=False;distribution_calls=0
    steps_invoked=0;steps_committed=0;started=time.monotonic();stop=None
    try:
        bridge=bridge_factory(root,profile,directory,started+ATTEMPT_SECONDS)
        before_tokenize=getattr(bridge,'evaluated',None)
        prompt_ids=bridge.tokenize(chatml(request),True,True)
        if getattr(bridge,'evaluated',None)!=before_tokenize:
            raise ValueError('TOKENIZE mutated evaluated counter')
        if (not isinstance(prompt_ids,list) or any(type(t) is not int or not 0<=t<bridge.vocabulary_size for t in prompt_ids)):
            raise ValueError('tokenizer returned invalid prompt IDs')
        if not prompt_ids or len(prompt_ids)+CAP > bridge.context_tokens:
            raise ValueError('exact prompt plus512 exceeds context')
        atomic_json(directory/'tokenized-prompt.json',{'request_sha256':request['request_sha256'],
            'prompt_ids':prompt_ids,'prompt_text_sha256':launch['prompt_text_sha256'],
            'context_tokens':bridge.context_tokens,'tokenizer_add_special':True,'tokenizer_parse_special':True})
        journal.append({'kind':'prompt_prepared','request_sha256':request['request_sha256'],
                        'prompt_ids':prompt_ids})
        prompt_invoked=True;bridge.start(prompt_ids)
        if getattr(bridge,'evaluated',None)!=len(prompt_ids):
            raise ValueError('PROMPT evaluated counter mismatch')
        prompt_committed=True
        journal.append({'kind':'prompt_committed','request_sha256':request['request_sha256']})
        for index in range(CAP):
            expected_evaluated=len(prompt_ids)+len(ids)
            if getattr(bridge,'evaluated',None)!=expected_evaluated:
                raise ValueError('native history differs before DIST')
            distribution_calls+=1
            logits=bridge.distribution()
            if getattr(bridge,'evaluated',None)!=expected_evaluated:
                raise ValueError('DIST mutated evaluated counter')
            if (len(logits)!=50 or any(type(t) is not int or not 0<=t<bridge.vocabulary_size
                    or t in profile['special_ids'] for t,_ in logits)):
                raise ValueError('masked top50 support differs')
            weights=quantize(logits,temperature=0.7,total=TOTAL)
            token,draw,draw_context=choose(weights,seed,request['request_sha256'],attempt_order,ids)
            intended_ids=ids+[token]
            intended_raw=bridge.render(intended_ids)
            if getattr(bridge,'evaluated',None)!=expected_evaluated:
                raise ValueError('TEXT mutated evaluated counter')
            if type(intended_raw) is not bytes: raise ValueError('TEXT must yield exact bytes')
            prepared=journal.append({'kind':'step_prepared','request_sha256':request['request_sha256'],
                'token_index':index,'prefix_ids':ids,'token_id':token,'weights':weights,
                'draw_uint24':draw,'draw_context_sha256':draw_context,
                'intended_raw_hex':intended_raw.hex(),'intended_raw_sha256':digest(intended_raw)})
            pending_step=True;steps_invoked+=1
            bridge.step(token)
            if getattr(bridge,'evaluated',None)!=expected_evaluated+1:
                raise ValueError('STEP evaluated counter mismatch')
            journal.append({'kind':'step_committed','prepared_sha256':prepared,'token_index':index,
                'token_id':token,'raw_sha256':digest(intended_raw)})
            pending_step=False;steps_committed+=1;ids=intended_ids;raw=intended_raw
            complete,stop_reason=completion(raw)
            if complete:
                stop=stop_reason
                break
        if stop is None: stop='max_tokens'
        # Only known committed raw reaches this parser; no convenience-file prefix is trusted.
        try:
            text=raw.decode('utf-8')
            accepted=parse_response(text)
            outcome='accepted';reason=None
        except (UnicodeDecodeError,ValueError,TypeError,RecursionError) as exc:
            outcome='shape_rejected';reason=type(exc).__name__+': '+str(exc)[:512]
        journal.append({'kind':'known_response','request_sha256':request['request_sha256'],
            'outcome':outcome,'stop':stop,'raw_sha256':digest(raw),'committed_tokens':len(ids)})
    except BaseException as exc:
        error={'type':type(exc).__name__,'detail':str(exc)[:1024]}
        # Conservative UNKNOWN even for metadata/process errors after attempt launch.
        outcome='unknown';reason='native_or_coordinator_failure_no_retry'
        try: journal.append({'kind':'unknown','request_sha256':request['request_sha256'],
            'error':error,'prompt_invoked':prompt_invoked,'prompt_committed':prompt_committed,
            'pending_step':pending_step,'steps_invoked':steps_invoked,'steps_committed':steps_committed})
        except BaseException: pass  # Torn/pending journal remains authoritative evidence.
    finally:
        cleanup_error=None
        if bridge is not None:
            try: bridge.close()
            except BaseException as exc:
                cleanup_error={'type':type(exc).__name__,'detail':str(exc)[:512]}
                outcome='unknown';reason='native_cleanup_failed_no_retry'
        journal.close()
    write_bytes(directory/'response.raw',raw)
    try: text=raw.decode('utf-8')
    except UnicodeDecodeError: text=None
    if text is not None: write_bytes(directory/'response.txt',raw)
    process=getattr(bridge,'process',None)
    result={'schema':VERSION+'-attempt-result','chunk_order':request['chunk_order'],
        'attempt_order':attempt_order,'request_sha256':request['request_sha256'],'seed_hex':seed,
        'outcome':outcome,'reason':reason,'stop_reason':stop,'raw_sha256':digest(raw),'raw_bytes':len(raw),
        'raw_text':text,'generated_ids':ids,'committed_output_tokens':len(ids),
        'prompt_token_count':len(prompt_ids),'prompt_invoked':prompt_invoked,'prompt_committed':prompt_committed,
        'distribution_calls_invoked':distribution_calls,'step_calls_invoked':steps_invoked,
        'step_calls_committed':steps_committed,'pending_step_unknown':pending_step,
        'provider_usage':None,'count_scope':'actual tokenizer length and bridge command counters; not API billing',
        'elapsed_seconds':time.monotonic()-started,'native_pid':getattr(process,'pid',None),
        'native_returncode_after_cleanup':None if process is None else process.poll(),
        'process_exit_scope':'bridge close normally terminates resident process; negative signal is not a worker failure',
        'accepted_words':accepted if outcome=='accepted' else None,
        'error':error,'cleanup_error':cleanup_error,'retry_allowed':False,
        'EOS_sampled':False,'EOS_scope':'EOS/specials masked; completion is JSON/cap not EOS',
        'semantic_equivalence_verified':False}
    atomic_json(directory/'result.json',result)
    return result


def validate_inputs(root):
    root=Path(root)
    protocol=json.loads((root/PROTOCOL).read_text());requests=json.loads((root/REQUESTS).read_text())
    profile=json.loads((root/PROFILE).read_text());validate_profile(profile)
    intent=json.loads((root/INTENT).read_text())
    if (intent.get('schema')!=VERSION+'-execution-intent' or intent.get('logical_requests')!=2
        or intent.get('maximum_physical_attempts')!=6 or intent.get('outcomes_known') is not False
        or intent.get('paid_calls_allowed') is not False or intent.get('closed_test_allowed') is not False
        or intent.get('protocol_sha256')!=sha_file(root/PROTOCOL)):
        raise ValueError('fixed execution intent differs')
    if (protocol['schema']!=VERSION+'-protocol' or protocol['token_cap']!=CAP or
        protocol['attempt_deadline_seconds']!=ATTEMPT_SECONDS or protocol['max_shape_attempts']!=3 or
        protocol['fixed_request_sha256']!=EXPECTED_REQUESTS or protocol['temperature']!=0.7 or
        protocol['profile']['path']!=PROFILE or protocol['profile']['sha256']!=sha_file(root/PROFILE) or
        protocol['requests']['path']!=REQUESTS or protocol['requests']['sha256']!=sha_file(root/REQUESTS)):
        raise ValueError('fixed native diagnostic protocol differs')
    if requests['schema']!=VERSION+'-requests' or len(requests['requests'])!=2:
        raise ValueError('exactly2 fixed logical requests required')
    for order,request in enumerate(requests['requests']): validate_request(request,order)
    return protocol,requests['requests'],profile


def verify_readback(root,readback_path):
    """Coordinator actual Git proof checked against entire fixed local closure."""
    root=Path(root);closure=json.loads((root/CLOSURE).read_text());receipt=json.loads(Path(readback_path).read_text())
    if (receipt.get('schema')!=VERSION+'-source-readback' or receipt.get('all_files_exact') is not True
        or not re.fullmatch('[a-f0-9]{40}',receipt.get('commit','')) or
        closure.get('schema')!=VERSION+'-source-closure'):
        raise ValueError('actual coordinator source readback required')
    if set(closure.get('files_sha256',{})) != REQUIRED_CLOSURE:
        raise ValueError('incomplete fixed source closure')
    expected={**closure['files_sha256'],CLOSURE:sha_file(root/CLOSURE)}
    if receipt.get('files_sha256')!=expected: raise ValueError('Git receipt set/hash mismatch')
    for relative,sha in expected.items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=relative or sha_file(root/p)!=sha:
            raise ValueError('frozen source/input changed: '+relative)
    return receipt


def run_campaign(root,output,readback_path,*,attempt_runner=run_attempt):
    root=Path(root);protocol,requests,profile=validate_inputs(root)
    receipt=verify_readback(root,readback_path)
    output=Path(output);output.mkdir(exist_ok=False,parents=True);fsync_directory(output.parent)
    atomic_json(output/'intent.json',{'schema':VERSION+'-campaign-launch','protocol':protocol,
        'source_readback':receipt,'requested_chunks':2,'physical_attempt_upper_bound':6,
        'outcomes_known':False,'heavy_NN_lease':'root coordinator exclusive'})
    states=[{'chunk_order':i,'request_sha256':r['request_sha256'],'status':'not_attempted','attempts':[]}
            for i,r in enumerate(requests)]
    atomic_json(output/'initial-denominator.json',{'chunks':states})
    terminal=False
    for chunk,request in zip(states,requests):
        if terminal:break
        for order in range(3):
            directory=output/f"chunk{chunk['chunk_order']}-attempt{order}"
            expected_identity={'schema':VERSION+'-attempt-result','chunk_order':chunk['chunk_order'],
                'request_sha256':request['request_sha256'],'attempt_order':order,
                'seed_hex':seed_hex(request['request_sha256'],order)}
            try:
                observed=attempt_runner(root,profile,request,order,directory)
                valid=(type(observed) is dict and all(type(observed.get(k)) is type(v) and observed.get(k)==v for k,v in expected_identity.items())
                       and observed.get('outcome') in {'accepted','shape_rejected','unknown'})
                if valid:
                    result=observed
                else:
                    try: canonical(observed); preserved=observed
                    except (ValueError,TypeError,RecursionError): preserved={'non_json_result_repr':repr(observed)[:4096]}
                    result={**expected_identity,'outcome':'unknown','reason':'attempt_result_contract_mismatch',
                            'observed_result':preserved,'retry_allowed':False}
            except BaseException as exc:
                # A missing result is UNKNOWN; never restart a reserved physical attempt.
                result={**expected_identity,'outcome':'unknown',
                        'error':{'type':type(exc).__name__,'detail':str(exc)[:1024]},
                        'missing_result_file':True,'retry_allowed':False}
            chunk['attempts'].append(result);chunk['status']=result['outcome']
            atomic_json(output/f"after-chunk{chunk['chunk_order']}-attempt{order}.json",{'chunks':states})
            if result['outcome']=='accepted':break
            if result['outcome']=='unknown':terminal=True;break
            if result['outcome']!='shape_rejected': raise ValueError('unregistered attempt outcome')
        if chunk['status']=='shape_rejected':chunk['status']='shape_attempts_exhausted'
    result={'schema':VERSION+'-campaign-result','chunks':states,'requested_chunks':2,
        'accepted_chunks':sum(s['status']=='accepted' for s in states),
        'attempts_executed':sum(len(s['attempts']) for s in states),'max_attempts':6,
        'status':'unknown_terminal' if terminal else 'complete_diagnostic',
        'full_cover_candidate_resource_complete':False,'original_fixed_transport_outcomes':0,
        'provider':'new EOS-masked Qwen JSON/cap diagnostic; not author GPT4o API',
        'closed_test_opened':False,'paid_calls':0,'automatic_retries':0}
    atomic_json(output/'result.json',result)
    return result
