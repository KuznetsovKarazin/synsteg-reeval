#!/usr/bin/env python3
"""Create exact native diagnostic source closure only; no model/tokenizer process."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit.author_candidate_local_native_v1 import (
    VERSION,REQUIRED_CLOSURE,validate_inputs,sha_file,atomic_json)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('new output only')
    protocol,requests,profile=validate_inputs(a.root)
    files={x:sha_file(a.root/x) for x in sorted(REQUIRED_CLOSURE)}
    value={'schema':VERSION+'-source-closure','files_sha256':files,
        'source_files':len(files),'model_calls':0,'tokenizer_calls':0,
        'status':'requires_actual_Git_readback_before_native_launch',
        'logical_chunks':2,'maximum_physical_attempts':6,'generation_outcomes_known':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,value)
    print(json.dumps({'source_files':len(files),'native_calls':0,'status':value['status']}))
if __name__=='__main__':main()
