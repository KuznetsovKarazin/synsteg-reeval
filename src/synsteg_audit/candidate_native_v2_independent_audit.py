"""Independent read-only postrun audit. No production verifier/native imports.

Recomputes evidence from saved bytes; never authenticates a writer able to alter
all trusted pins. Native-origin claims are limited to recorded metadata.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from .candidate_native_v2_audit_math import prefix_state, assess_format, replay_sample

ID = 'author-candidate-local-native-v2'
AUDIT = 'author-candidate-native-v2-independent-audit'
GUARD = 'candidate-journal-guard-v2'
BASE = 'experiments/' + ID
OUTPUT = 'results/' + ID + '/native-diagnostic-v2'
PROFILE = 'experiments/framed-interval-public-v1/native-public-profile.json'
REQUESTS = 'experiments/author-candidate-format-v2/requests.json'
PROTOCOL = BASE + '/protocol.json'
CLOSURE = BASE + '/source-closure.json'
PINNED = {
    PROFILE: 'f32097750dc898fce418a29146587a4229eae1dcb76e3cbcd374b692846adc8a',
    REQUESTS: '0cc5750f8251cb7b10bb29d00ec858d5b005d0df42fb7d2781152c807a4ae421',
    PROTOCOL: '737c82054a7d3060d1b8295d972dd9e23b3d53d0292fdfa9f12df3fdfad012ea',
    CLOSURE: 'ceb4c2957d0a98f9248eaad199594df22242bfcd6662906a2fa2a737070dabe0',
}
REQUEST_HASHES = ('15f413e60449231fe505a45091d838f36155d15307bcdf16b27d1215ffa05104',
                  'd9f3597b716fb4fbb888a83ceccd881f94635537132ac5dc00c0748ddd14b4f4')
RESOURCE_ROOT = 'resources/seeded-pair-llm-v1/'
BRIDGE = 'seeded_pair_llama_bridge'
MODEL = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
CPU = 'runtime/llama-b11791/libggml-cpu-haswell.so'
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 160

class AuditError(ValueError):
    pass

def need(test, reason):
    if not test:
        raise AuditError(reason)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def same(a, b):
    return canonical(a) == canonical(b)

def closed(value, keys):
    need(type(value) is dict and set(value) == set(keys), 'closed object schema differs')

def uint(value, maximum=2**31-1):
    return type(value) is int and 0 <= value <= maximum

def hash_string(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None

def parse_json(raw, *, exact=False):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def no_constant(_):
        raise AuditError('nonfinite JSON')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=no_constant)
        if exact:
            need(canonical(value) == raw, 'noncanonical JSON')
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise AuditError('invalid JSON bytes') from exc

def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

def read_regular(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        first = os.fstat(fd)
        need(stat.S_ISREG(first.st_mode) and first.st_size <= MAX_BYTES, 'nonregular/oversized evidence')
        chunks = []; total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block); need(total <= MAX_BYTES, 'growing evidence exceeds bound'); chunks.append(block)
        need(identity(first) == identity(os.fstat(fd)) == identity(os.lstat(path)), 'evidence changed during read')
        return b''.join(chunks)
    finally:
        os.close(fd)

def snapshot(directory):
    """Exact recursive regular-file inventory; rejects links/devices/FIFOs."""
    directory = Path(directory)
    need(stat.S_ISDIR(os.lstat(directory).st_mode), 'campaign must be a real directory')
    data = {}; dirs = []
    def walk(path, depth):
        need(depth < 6, 'unexpected directory depth')
        before = os.lstat(path); dirs.append(path.relative_to(directory).as_posix())
        for item in sorted(os.scandir(path), key=lambda x:x.name):
            child = Path(item.path); mode = item.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                walk(child, depth+1)
            else:
                need(stat.S_ISREG(mode), 'nonregular campaign member: '+child.name)
                key = child.relative_to(directory).as_posix(); data[key] = read_regular(child)
                need(len(data) <= MAX_FILES, 'too many campaign files')
        need(identity(before) == identity(os.lstat(path)), 'directory changed during inventory')
    walk(directory, 0)
    return data, sorted(dirs)

def inventory(data):
    return {name:{'bytes':len(raw), 'sha256':sha(raw)} for name, raw in sorted(data.items())}

def binding(profile, request, order, protocol_hash):
    return {'experiment_id':ID, 'attempt_id':f"chunk{request['chunk_order']}-attempt{order}",
            'request_sha256':request['request_sha256'], 'resource_sha256':sha(canonical(profile)),
            'protocol_sha256':protocol_hash, 'token_cap':512}

def journal(raw, expected, profile, request, order, *, complete):
    """Independent finite-state replay, including every prepared sample."""
    need(type(raw) is bytes and raw and len(raw) <= MAX_BYTES, 'missing journal')
    torn = not raw.endswith(b'\n'); rows = raw.splitlines(keepends=True)
    if torn:
        need(not complete, 'torn journal cannot be complete'); rows = rows[:-1]
    state = {'rows':0, 'head':'0'*64, 'prompt_ids':[], 'prompt_committed':False,
             'ids':[], 'raw':b'', 'pending':None, 'terminal':False, 'stop':None,
             'prepared_steps':0, 'torn_last_line':torn, 'first_stop':None}
    for index, line in enumerate(rows):
        row = parse_json(line[:-1], exact=True)
        closed(row, ('sequence','previous_sha256','event','event_sha256'))
        need(uint(row['sequence']) and row['sequence'] == index, 'journal sequence')
        need(row['previous_sha256'] == state['head'], 'journal previous hash')
        body = {k:row[k] for k in ('sequence','previous_sha256','event')}
        need(hash_string(row['event_sha256']) and sha(canonical(body)) == row['event_sha256'], 'journal hash')
        e = row['event']; need(type(e) is dict and type(e.get('kind')) is str, 'event object')
        need(not state['terminal'], 'event after trace terminal')
        kind = e['kind']; pending = state['pending']
        if index == 0:
            closed(e, ('kind','binding_sha256'))
            need(kind == 'opened' and e['binding_sha256'] == sha(canonical(expected)), 'initial binding')
        elif kind == 'prompt_prepared':
            closed(e, ('kind','prompt_ids'))
            need(index == 1 and not state['prompt_ids'] and pending is None, 'prompt ordering')
            ids = e['prompt_ids']
            need(type(ids) is list and 0 < len(ids) and len(ids)+512 <= profile['ready']['context_tokens']
                 and all(uint(i, profile['ready']['vocabulary_size']-1) for i in ids), 'prompt/context IDs')
            state['prompt_ids'] = ids; state['pending'] = (kind, row['event_sha256'], e)
        elif kind == 'prompt_committed':
            closed(e, ('kind','prepared_sha256','evaluated_tokens'))
            need(pending is not None and pending[:2] == ('prompt_prepared',e['prepared_sha256']), 'prompt acknowledgement')
            need(type(e['evaluated_tokens']) is int and e['evaluated_tokens'] == len(state['prompt_ids']), 'prompt count')
            state['prompt_committed'] = True; state['pending'] = None
        elif kind == 'step_prepared':
            closed(e, ('kind','token_index','token_id','intended_raw_hex','sample_evidence'))
            need(state['prompt_committed'] and pending is None and state['first_stop'] is None, 'STEP ordering/continued after first stop')
            need(type(e['token_index']) is int and e['token_index'] == len(state['ids']) < 512, 'STEP index/cap')
            need(type(e['intended_raw_hex']) is str and len(e['intended_raw_hex']) <= 8*1024*1024
                 and re.fullmatch(r'(?:[0-9a-f]{2})*',e['intended_raw_hex']) is not None, 'STEP raw hex')
            replay_sample(e['sample_evidence'],e['token_id'],state['ids'],request['request_sha256'],order,profile)
            state['pending'] = (kind,row['event_sha256'],e); state['prepared_steps'] += 1
        elif kind == 'step_committed':
            closed(e, ('kind','prepared_sha256','evaluated_tokens'))
            need(pending is not None and pending[:2] == ('step_prepared',e['prepared_sha256']), 'STEP acknowledgement')
            need(type(e['evaluated_tokens']) is int and e['evaluated_tokens'] == len(state['prompt_ids'])+len(state['ids'])+1, 'STEP count')
            state['ids'].append(pending[2]['token_id']); state['raw'] = bytes.fromhex(pending[2]['intended_raw_hex']); state['pending'] = None
            current = prefix_state(state['raw'])
            if current != 'incomplete':state['first_stop'] = current
        elif kind == 'trace_complete':
            closed(e, ('kind','output_ids','response_sha256','response_bytes','stop_label'))
            need(state['prompt_committed'] and pending is None and state['ids'], 'uncommitted/empty terminal')
            need(same(e['output_ids'],state['ids']) and type(e['response_bytes']) is int
                 and e['response_bytes']==len(state['raw']) and e['response_sha256']==sha(state['raw']), 'terminal bytes/IDs')
            stop = state['first_stop'] or ('token_cap' if len(state['ids'])==512 else None)
            need(stop is not None and e['stop_label']==stop, 'first-stop law')
            state['terminal']=True;state['stop']=stop
        else:
            raise AuditError('unexpected event kind')
        state['rows'] += 1;state['head']=row['event_sha256']
    if complete:
        need(state['terminal'] and state['pending'] is None and not torn, 'incomplete trace')
    return state

def seal_check(data, name, expected, profile, request, order, commitment):
    prefix = name+'/sealed/'
    members = {k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)}
    required = {'writer.lock','binding.json','journal.jsonl','response.raw','result.json','terminal.json'}
    need(set(members)==required, 'sealed inventory differs')
    need(members['binding.json']==canonical(expected), 'bundle binding bytes')
    need(members['writer.lock']==canonical({'schema':GUARD+'-lease','binding_sha256':sha(canonical(expected))}), 'writer lease')
    need(hash_string(commitment['terminal_sha256']) and sha(members['terminal.json'])==commitment['terminal_sha256'], 'external terminal pin')
    trace = journal(members['journal.jsonl'],expected,profile,request,order,complete=True)
    need(members['response.raw']==trace['raw'], 'response vs journal')
    result={'schema':GUARD+'-result','status':'trace_complete','binding_sha256':sha(canonical(expected)),
            'output_ids':trace['ids'],'committed_output_tokens':len(trace['ids']),'prompt_tokens':len(trace['prompt_ids']),
            'response_bytes':len(trace['raw']),'response_sha256':sha(trace['raw']),
            'journal_bytes':len(members['journal.jsonl']),'journal_sha256':sha(members['journal.jsonl']),
            'journal_rows':trace['rows'],'journal_head_sha256':trace['head'],'stop_label':trace['stop'],
            'shape_or_semantics_accepted':False}
    need(members['result.json']==canonical(result), 'sealed result bytes')
    manifest={'schema':GUARD+'-terminal','binding_sha256':sha(canonical(expected)),
              'files':inventory({k:v for k,v in members.items() if k!='terminal.json'}),
              'journal_head_sha256':trace['head'],'journal_rows':trace['rows']}
    need(members['terminal.json']==canonical(manifest), 'terminal manifest canonical bytes/types')
    return trace

def process_identity(x):
    closed(x, ('namespace_pids','pid_namespace','proc_pid','proc_ppid','starttime_ticks'))
    need(type(x['namespace_pids']) is list and x['namespace_pids'] and all(uint(p) and p>0 for p in x['namespace_pids']), 'namespace PIDs')
    need(type(x['pid_namespace']) is str and re.fullmatch(r'pid:\[[0-9]+\]',x['pid_namespace']) is not None, 'PID namespace')
    need(all(uint(x[k]) and x[k]>0 for k in ('proc_pid','proc_ppid','starttime_ticks')), 'process counters')
    need(x['namespace_pids'][0]==x['proc_pid'], 'visible process identity')

def native_evidence(data, name, profile, result):
    prefix=name+'/native/';files={k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)}
    need(set(files)=={'resource-check.json','native-ready.json','proc-identity-capture.json','native.stderr.log'}, 'native evidence inventory')
    resource=parse_json(files['resource-check.json']);ready=parse_json(files['native-ready.json']);capture=parse_json(files['proc-identity-capture.json'])
    pd=sha(canonical(profile))
    need(same(resource,{'schema':'framed-interval-public-v1-native-resource-check','verified_files':len(profile['resource_files']),
         'public_profile_sha256':pd,'model_calls':0,'tokenizer_calls':0,
         'scope':'current files/runtime only; no historical or resident attestation'}), 'resource verification record')
    closed(ready,('ready','resident','resource_profile_sha256','prompt_calls','step_calls'))
    need(same(ready['ready'],profile['ready']) and ready['resource_profile_sha256']==pd
         and type(ready['prompt_calls']) is int and ready['prompt_calls']==0
         and type(ready['step_calls']) is int and ready['step_calls']==0, 'READY/resource binding')
    closed(capture,('schema','popen_pid','parent','child','executable','public_profile_sha256'))
    need(capture['schema']=='framed-interval-public-v1-owned-process' and capture['public_profile_sha256']==pd, 'captured process schema/profile')
    process_identity(capture['child']);process_identity(capture['parent'])
    need(type(capture['popen_pid']) is int and capture['popen_pid']>0 and capture['popen_pid']==capture['child']['namespace_pids'][-1], 'Popen namespace PID')
    need(capture['child']['proc_ppid']==capture['parent']['proc_pid']
         and capture['child']['pid_namespace']==capture['parent']['pid_namespace'], 'owned parent-child relation')
    executable=capture['executable'];closed(executable,('bytes','device','inode','path','sha256'))
    bridge_item=profile['resource_files'][BRIDGE]
    need(type(executable['bytes']) is int and executable['bytes']==bridge_item['bytes'] and executable['sha256']==bridge_item['sha256'], 'native executable hash/bytes')
    need(uint(executable['device']) and uint(executable['inode'],2**63-1), 'executable inode/device')
    suffix='/'+RESOURCE_ROOT+BRIDGE
    need(type(executable['path']) is str and executable['path'].startswith('/') and executable['path'].endswith(suffix), 'native executable path')
    resource_base=executable['path'][:-len(BRIDGE)]
    resident=ready['resident'];closed(resident,('status','pid','mapped_files_sha256','identity','scope'))
    need(resident['status']=='passed' and type(resident['pid']) is int and resident['pid']==capture['popen_pid'], 'resident PID')
    ident=resident['identity'];closed(ident,('schema','parent','before','after','maps_sha256','pidfd_identity','pidfd_live_before_and_after_maps','popen_pid','proc_visible_pid','scope'))
    need(ident['schema']=='proc-namespace-identity-v2' and ident['pidfd_live_before_and_after_maps'] is True
         and hash_string(ident['maps_sha256']), 'resident capture contract')
    expected_snapshot={'executable':executable,'identity':capture['child']}
    need(same(ident['before'],expected_snapshot) and same(ident['after'],expected_snapshot)
         and same(ident['parent'],capture['parent']), 'resident identity drift')
    need(type(ident['popen_pid']) is int and ident['popen_pid']==capture['popen_pid']
         and type(ident['proc_visible_pid']) is int and ident['proc_visible_pid']==capture['child']['proc_pid'], 'resident visible/Popen PIDs')
    pidfd=ident['pidfd_identity'];closed(pidfd,('inode','namespace_pids','proc_pid'))
    need(uint(pidfd['inode'],2**63-1) and pidfd['inode']>0 and same(pidfd['namespace_pids'],capture['child']['namespace_pids'])
         and type(pidfd['proc_pid']) is int and pidfd['proc_pid']==capture['child']['proc_pid'], 'pidfd identity')
    mapped=resident['mapped_files_sha256'];need(type(mapped) is dict and mapped, 'mapped file evidence')
    resources=set()
    for path,digest in mapped.items():
        need(type(path) is str and hash_string(digest) and path.startswith('/') and '..' not in PurePosixPath(path).parts, 'mapped file path/digest')
        if path.startswith(resource_base):
            relative=path[len(resource_base):];need(relative in profile['resource_files'], 'unregistered mapped resource')
            need(digest==profile['resource_files'][relative]['sha256'], 'mapped resource hash')
            need('libggml-cpu-' not in relative or relative==CPU, 'unexpected CPU backend')
            resources.add(relative)
        else:
            need(profile['system_libraries_sha256'].get(path)==digest, 'unregistered mapped system bytes')
    need({BRIDGE,MODEL,CPU}.issubset(resources) and any(Path(p).name.startswith('libllama.so') for p in resources), 'required resident resources missing')
    need(type(result['native_pid']) is int and result['native_pid']==capture['popen_pid'], 'attempt PID')
    need(type(result['native_returncode_after_cleanup']) is int and result['native_returncode_after_cleanup'] in (0,-15,-9), 'native process not observably reaped normally/terminated')
    return {'recorded_native_identity_consistent':True,'pid':capture['popen_pid'],
            'pid_namespace':capture['child']['pid_namespace'],'proc_visible_pid':capture['child']['proc_pid'],
            'starttime_ticks':capture['child']['starttime_ticks'],'mapped_resource_count':len(resources),
            'native_stderr_bytes':len(files['native.stderr.log']),'native_stderr_sha256':sha(files['native.stderr.log']),
            'model_origin_authenticated':False,'raw_maps_available':False}

def validate_source(root, launch):
    raw={p:read_regular(root/p) for p in PINNED}
    for p,expected in PINNED.items():need(sha(raw[p])==expected,'fixed source binding changed: '+p)
    profile=parse_json(raw[PROFILE]);requests=parse_json(raw[REQUESTS])['requests'];protocol=parse_json(raw[PROTOCOL]);closure=parse_json(raw[CLOSURE])
    need(len(requests)==2,'request denominator')
    for i,r in enumerate(requests):
        need(type(r['chunk_order']) is int and r['chunk_order']==i and r['request_sha256']==REQUEST_HASHES[i]
             and sha(canonical({k:v for k,v in r.items() if k!='request_sha256'}))==REQUEST_HASHES[i], 'fixed request hash')
    closed(launch,('schema','protocol','source_readback','outcomes_known','requested_chunks','maximum_physical_attempts'))
    need(launch['schema']==ID+'-campaign-launch' and same(launch['protocol'],protocol)
         and launch['outcomes_known'] is False and type(launch['requested_chunks']) is int and launch['requested_chunks']==2
         and type(launch['maximum_physical_attempts']) is int and launch['maximum_physical_attempts']==6,'campaign launch contract')
    receipt=launch['source_readback'];need(type(receipt) is dict and receipt.get('schema')==ID+'-source-readback'
         and receipt.get('all_files_exact') is True and hash_string(receipt.get('source_commit'),40)
         and hash_string(receipt.get('activation_commit'),40),'source readback receipt')
    files=receipt.get('files_sha256');expected={**closure['files_sha256'],CLOSURE:PINNED[CLOSURE]}
    need(type(files) is dict and all(files.get(p)==v for p,v in expected.items()),'source closure receipt')
    extra=set(files)-set(expected);need(len(extra)==2,'enabled intent/activation exact set')
    added={}
    for path,digest in files.items():
        need(type(path) is str and not path.startswith('/') and '..' not in PurePosixPath(path).parts and hash_string(digest),'source receipt path')
        r=read_regular(root/path);need(sha(r)==digest,'actual source differs from receipt: '+path)
        if path in extra:
            need(path.startswith(BASE+'/'),'enabled metadata outside experiment');added[path]=parse_json(r)
    need(all(type(v) is dict for v in added.values()),'activation/intent metadata must be objects')
    intents=[(p,v) for p,v in added.items() if v.get('schema')==ID+'-execution-intent']
    activations=[(p,v) for p,v in added.items() if v.get('schema')==ID+'-activation']
    need(len(intents)==len(activations)==1,'distinct activation and intent')
    p,intent=intents[0]
    expected_intent={'schema':ID+'-execution-intent','run_id':'native-diagnostic-v2','protocol_sha256':PINNED[PROTOCOL],
        'source_closure_sha256':PINNED[CLOSURE],'output':OUTPUT,'logical_requests':2,'maximum_physical_attempts':6,
        'outcomes_known':False,'automatic_retry':False,'paid_calls_allowed':False,'closed_test_allowed':False}
    need(same(intent,expected_intent),'execution intent contract')
    expected_activation={'schema':ID+'-activation','enabled':True,'protocol_sha256':PINNED[PROTOCOL],
        'source_closure_sha256':PINNED[CLOSURE],'execution_intent_sha256':files[p],'output':OUTPUT,'logical_requests':2,
        'maximum_physical_attempts':6,'root_exclusive_nn_lease_required':True,'paid_calls_allowed':False,'closed_test_allowed':False}
    need(same(activations[0][1],expected_activation),'enabled activation contract')
    return profile,requests,receipt

def attempt(data, name, expected, profile, request, order, runtime, source_receipt):
    def read(relative,exact=True):return parse_json(data[relative],exact=exact)
    reservation=read('reservations/'+name+'.json')
    need(same(reservation,{'schema':ID+'-attempt-reservation','binding':expected,
        'source_commit':source_receipt['source_commit'],'activation_commit':source_receipt['activation_commit'],
        'no_retry':True,'outcome_known':False}),'attempt reservation')
    need(type(runtime) is dict, 'runtime attempt object')
    if runtime.get('schema')==ID+'-coordinator-unknown':
        need(runtime.get('outcome')=='unknown' and same(runtime.get('binding'),expected)
             and runtime.get('retry_allowed') is False,'coordinator UNKNOWN')
        return {'runtime_outcome':'unknown','evidence_status':'runtime_UNKNOWN','known_format_verified':False,
                'coordinator_unknown':True,'committed_tokens':None}
    result=read(name+'/attempt-result.json')
    need(same(result,runtime),'campaign/attempt persisted result differs')
    seed=sha(canonical([ID,'fixed-public-diagnostic-seed',request['request_sha256'],order]))
    for key,val in {'schema':ID+'-attempt-result','binding':expected,'chunk_order':request['chunk_order'],'attempt_order':order,
                    'request_sha256':request['request_sha256'],'seed_hex':seed}.items():need(same(result.get(key),val),'attempt identity '+key)
    need(result.get('outcome') in {'unknown','format_accepted','format_rejected','parser_resource_refusal'},'runtime outcome')
    text='<|im_start|>system\n'+request['system']+'<|im_end|>\n<|im_start|>user\n'+request['user']+'<|im_end|>\n<|im_start|>assistant\n'
    expected_launch={'schema':ID+'-attempt-launch','binding':expected,'seed_hex':seed,'request':request,
        'prompt_text_sha256':sha(text.encode()),'token_cap':512,'attempt_seconds':600}
    need(same(read(name+'/launch.json'),expected_launch),'attempt launch')
    summary={'runtime_outcome':result['outcome'],'evidence_status':'runtime_UNKNOWN' if result['outcome']=='unknown' else 'verified_known',
             'known_format_verified':False,'coordinator_unknown':False,'committed_tokens':None}
    if result['outcome']=='unknown':
        jkey=name+'/sealed/journal.jsonl'
        if jkey in data:
            try:
                partial=journal(data[jkey],expected,profile,request,order,complete=False)
                summary.update(committed_tokens=len(partial['ids']),pending_prepared=partial['pending'] is not None,
                               torn_last_line=partial['torn_last_line'],journal_partial_structural_check=True)
            except (ValueError,TypeError,KeyError) as exc:
                summary['partial_journal_issue']=type(exc).__name__+': '+str(exc)
        return summary
    commitment=read('commitments/'+name+'.json')
    closed(commitment,('schema','binding','terminal_sha256','source','git_output_readback_completed'))
    need(same(commitment,{'schema':ID+'-terminal-commitment','binding':expected,'terminal_sha256':commitment.get('terminal_sha256'),
        'source':'in-memory seal return persisted outside attempt bundle','git_output_readback_completed':False}),'external commitment schema')
    root_files={key[len(name)+1:] for key in data if key.startswith(name+'/') and '/' not in key[len(name)+1:]}
    need(root_files=={'launch.json','tokenized-prompt.json','observed-response.raw','attempt-result.json'}, 'known attempt root inventory')
    trace=seal_check(data,name,expected,profile,request,order,commitment)
    need(same(read(name+'/tokenized-prompt.json'),{'prompt_ids':trace['prompt_ids'],'request_sha256':request['request_sha256'],
        'add_special':True,'parse_special':True}),'saved prompt binding')
    need(data[name+'/observed-response.raw']==trace['raw'],'observed response bytes')
    assessed=assess_format(trace['raw'],request['words']);count=len(trace['ids'])
    expected_fields={'outcome':assessed['outcome'],'format_result':assessed['format_result'],
        'raw_sha256':sha(trace['raw']),'raw_bytes':len(trace['raw']),'generated_ids':trace['ids'],
        'committed_output_tokens':count,'step_calls_committed':count,'step_calls_invoked':count,
        'distribution_calls_invoked':count,'prompt_token_count':len(trace['prompt_ids']),'prompt_invoked':True,'prompt_committed':True,
        'pending_step_unknown':False,'stop_reason':trace['stop'],'terminal_sha256':commitment['terminal_sha256'],
        'commitment_sha256':sha(data['commitments/'+name+'.json']),'error':None,'cleanup_error':None,
        'usable_candidate_rows':None,'capacity_bits':None,'semantic_equivalence_verified':False,'provider_usage':None,
        'retry_allowed':False,'EOS_sampled':False}
    for key,val in expected_fields.items():need(same(result.get(key),val),'known result differs: '+key)
    need(type(result.get('prefix')) is dict and result['prefix'].get('state')==assessed['prefix_state'],'prefix state')
    need(type(result.get('elapsed_seconds')) in (int,float) and 0<=result['elapsed_seconds']<3600,'elapsed metadata')
    native=native_evidence(data,name,profile,result)
    summary.update(known_format_verified=True,committed_tokens=count,format_result=assessed['format_result'],
        stop_reason=trace['stop'],terminal_sha256=commitment['terminal_sha256'],raw_sha256=sha(trace['raw']),
        raw_bytes=len(trace['raw']),native_evidence=native)
    return summary

def audit_data(root,data,dirs,external_receipt=None):
    """Deterministic auditor over one stable inventory; errors preserve denominator."""
    issues=[];attempts=[];chunks=[];runtime=None;source_ok=False;terminal_pins={};expected_names=[]
    def issue(where,exc):issues.append({'where':where,'reason':type(exc).__name__+': '+str(exc)})
    try:
        launch=parse_json(data['intent.json'],exact=True)
        profile,requests,source_receipt=validate_source(root,launch);source_ok=True
        runtime=parse_json(data['result.json'],exact=True)
        initial={'chunks':[{'chunk_order':i,'request_sha256':r['request_sha256'],'status':'not_attempted','attempts':[]} for i,r in enumerate(requests)],
                 'full_cover_chunks':109,'outside_diagnostic_chunks':107}
        need(same(parse_json(data['initial-denominator.json'],exact=True),initial),'initial denominator')
        need(type(runtime) is dict and runtime.get('schema')==ID+'-campaign-result','campaign result schema')
        logical=runtime.get('chunks');need(type(logical) is list and len(logical)==2,'two logical chunks required')
        halt=None;accepted=0;physical=0;native_identities=set();states=initial['chunks']
        for c,request in zip(logical,requests):
            closed(c,('chunk_order','request_sha256','status','attempts'))
            need(type(c['chunk_order']) is int and c['chunk_order']==request['chunk_order'] and c['request_sha256']==request['request_sha256'],'chunk identity')
            records=c['attempts'];need(type(records) is list and len(records)<=3,'attempt count')
            need(not halt or not records,'attempt after global halt')
            need(records or halt is not None,'unattempted chunk without terminal cause')
            last=None
            for order,r in enumerate(records):
                name=f"chunk{c['chunk_order']}-attempt{order}";expected_names.append(name);physical+=1
                need(last in (None,'format_rejected'),'retry after accepted/unknown/resource')
                expected=binding(profile,request,order,PINNED[PROTOCOL]);row={'attempt':name,'chunk_order':c['chunk_order'],'attempt_order':order,
                    'runtime_outcome':r.get('outcome') if type(r) is dict else None,'evidence_status':'evidence_UNKNOWN','known_format_verified':False}
                try:
                    row.update(attempt(data,name,expected,profile,request,order,r,source_receipt))
                    if row.get('partial_journal_issue'):raise AuditError(row['partial_journal_issue'])
                    if row.get('terminal_sha256'):terminal_pins[name]=row['terminal_sha256']
                    if row.get('native_evidence'):
                        ne=row['native_evidence'];identity_key=(ne['pid_namespace'],ne['proc_visible_pid'],ne['starttime_ticks'])
                        need(identity_key not in native_identities,'cold process identity reused');native_identities.add(identity_key)
                except (ValueError,TypeError,KeyError,OSError) as exc:issue(name,exc)
                attempts.append(row)
                last=r.get('outcome') if type(r) is dict else None
                need(last in {'format_accepted','format_rejected','parser_resource_refusal','unknown'},'attempt outcome enum')
                states[c['chunk_order']]['attempts'].append(r);states[c['chunk_order']]['status']=last
                need(same(parse_json(data['after-'+name+'.json'],exact=True),{'chunks':states}),'after-attempt checkpoint differs')
            derived='not_attempted' if not records else last
            if last=='format_rejected':need(len(records)==3,'premature exhaustion');derived='format_attempts_exhausted'
            need(c['status']==derived,'chunk status not derived');states[c['chunk_order']]['status']=derived
            if derived=='format_accepted':accepted+=1
            if derived in {'unknown','parser_resource_refusal'}:halt=derived
            chunks.append({'chunk_order':c['chunk_order'],'runtime_status':derived,'attempts_reported':len(records)})
        expected_top={'requested_chunks':2,'format_accepted_chunks_observed':accepted,
            'complete_campaign_format_accepted_chunks':None if halt=='unknown' else accepted,'attempts_executed':physical,'max_attempts':6,
            'status':halt+'_terminal' if halt else 'complete_diagnostic_pending_Git_audit','full_cover_chunks':109,'outside_diagnostic_chunks':107,
            'full_cover_candidate_resource_complete':False,'original_fixed_transport_outcomes':0,'usable_candidate_rows':None,
            'capacity_bits':None,'semantic_equivalence_verified':False,'paid_calls':0,'closed_test_opened':False,'automatic_retries':0,
            'post_save_Git_evidence_verified':False}
        for key,value in expected_top.items():need(same(runtime.get(key),value),'campaign field '+key)
        present_names={p.split('/')[0] for p in data if re.match(r'^chunk[0-9]+-attempt[0-9]+(?:/|$)',p)}
        need(present_names.issubset(set(expected_names)),'unregistered attempt files')
        reservations={p[len('reservations/'):-5] for p in data if p.startswith('reservations/') and p.endswith('.json')}
        need(reservations==set(expected_names) and {p for p in data if p.startswith('reservations/')}=={'reservations/'+n+'.json' for n in expected_names},'reservation denominator mismatch')
        commitments={p[len('commitments/'):-5] for p in data if p.startswith('commitments/') and p.endswith('.json')}
        need(commitments.issubset(set(expected_names)) and {p for p in data if p.startswith('commitments/')}=={'commitments/'+n+'.json' for n in commitments},'unregistered commitment')
        tops={p.split('/')[0] for p in data}
        allowed={'intent.json','initial-denominator.json','result.json','reservations','commitments'}|set(expected_names)|{'after-'+n+'.json' for n in expected_names}
        need(tops.issubset(allowed),'unexpected campaign top-level artifact')
        need({'.','reservations','commitments'}.issubset(set(dirs)), 'required campaign directories')
        need(set(dirs).issubset({'.','reservations','commitments'}|set(expected_names)|{n+'/sealed' for n in expected_names}|{n+'/native' for n in expected_names}), 'unexpected directory')
    except (ValueError,TypeError,KeyError,OSError) as exc:issue('campaign',exc)
    seen={x['attempt'] for x in attempts}
    for d in dirs:
        if re.fullmatch(r'chunk[0-9]+-attempt[0-9]+',d) and d not in seen:
            observed=None
            try:
                if d+'/attempt-result.json' in data:observed=parse_json(data[d+'/attempt-result.json']).get('outcome')
            except (ValueError,TypeError,AttributeError):pass
            attempts.append({'attempt':d,'runtime_outcome':observed,'evidence_status':'evidence_UNKNOWN',
                             'known_format_verified':False,'reason':'Unvalidated retained attempt after campaign-level failure.'})
    external_ok=False
    if external_receipt is not None:
        try:
            closed(external_receipt,('schema','all_files_exact','output_commit','campaign_files','terminal_pins'))
            need(external_receipt['schema']==AUDIT+'-output-readback' and external_receipt['all_files_exact'] is True
                 and hash_string(external_receipt['output_commit'],40),'external Git receipt identity')
            need(same(external_receipt['campaign_files'],inventory(data)),'external output inventory differs')
            need(same(external_receipt['terminal_pins'],terminal_pins),'external terminal pins differ')
            external_ok=True
        except (ValueError,TypeError,KeyError) as exc:issue('external_output_readback',exc)
    unknown_runtime=any(x.get('runtime_outcome')=='unknown' for x in attempts)
    accepted_verified=sum(x.get('known_format_verified') and x.get('runtime_outcome')=='format_accepted' for x in attempts)
    runtime_object=runtime if type(runtime) is dict else {}
    result={'schema':AUDIT+'-report','source_contract_verified':source_ok,'issues':issues,'attempts':attempts,'chunks':chunks,
        'requested_chunks':2,'full_cover_chunks':109,'outside_diagnostic_chunks':107,'maximum_physical_attempts':6,
        'runtime_status':runtime_object.get('status'),'runtime_reported_attempts':runtime_object.get('attempts_executed'),
        'reservation_files_present':sum(p.startswith('reservations/') and p.endswith('.json') for p in data),
        'attempt_directories_present':sorted(d for d in dirs if re.fullmatch(r'chunk[0-9]+-attempt[0-9]+',d)),
        'observed_verified_format_accepted_attempts':accepted_verified,
        'complete_verified_format_accepted_chunks':None if issues or unknown_runtime else accepted_verified,
        'assessment':'evidence_UNKNOWN' if issues else ('runtime_UNKNOWN' if unknown_runtime else ('verified_recorded_evidence_after_Git_readback' if external_ok else 'verified_local_evidence_pending_Git_readback')),
        'post_save_Git_evidence_verified':external_ok,'campaign_files':inventory(data),'terminal_pins':terminal_pins,
        'candidate_usable_rows':None,'capacity_bits':None,'semantic_equivalence_verified':False,'original_fixed_transport_outcomes':0,
        'audit_native_model_calls':0,'audit_tokenizer_calls':0,'audit_codec_calls':0,
        'limitations':['No model/logit origin authentication; saved native identity metadata is checked for consistency only.',
          'The source/output readback receipts are trusted external observations, not authenticated by this offline auditor.',
          'Hash pins and advisory locks do not provide OS immutability or hardware attestation.',
          'No tokenizer replay, no raw /proc maps reconstruction, no unseen generation, and no semantic equivalence test.']}
    return result

def audit_campaign(root, campaign, *, external_receipt=None):
    root=Path(root).resolve();campaign=Path(campaign)
    need(campaign.resolve()==root/OUTPUT,'fixed candidate-v2 campaign path required')
    fd=os.open(root/'results/project-nn.lock',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        need(stat.S_ISREG(os.fstat(fd).st_mode),'NN lease must be regular')
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        before=os.fstat(fd);need(identity(before)==identity(os.lstat(root/'results/project-nn.lock')),'NN lease replaced')
        data,dirs=snapshot(campaign);result=audit_data(root,data,dirs,external_receipt)
        later,later_dirs=snapshot(campaign)
        need(inventory(data)==inventory(later) and dirs==later_dirs,'campaign changed during audit')
        need(identity(before)==identity(os.fstat(fd))==identity(os.lstat(root/'results/project-nn.lock')),'NN lease changed during audit')
        return result
    finally:os.close(fd)
