#!/usr/bin/env python3
"""Explicit coordinator entrypoint; importing this file never starts native code."""
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit.author_candidate_local_native_v1 import run_campaign

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--readback',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required after actual source freeze/readback and exclusive NN lease')
    result=run_campaign(args.root,args.output,args.readback)
    print(json.dumps({'status':result['status'],'accepted_chunks':result['accepted_chunks'],
        'requested_chunks':2,'attempts_executed':result['attempts_executed'],
        'full_cover_candidate_resource_complete':False},sort_keys=True))
    return 0 if result['status']=='complete_diagnostic' else 2

if __name__=='__main__':raise SystemExit(main())
