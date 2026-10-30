#!/usr/bin/env python3
"""Read saved native-v2 evidence only; no NN/tokenizer/codec or production verifier."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit.candidate_native_v2_independent_audit import audit_campaign,read_regular,parse_json,OUTPUT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--campaign',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--external-output-readback',type=Path)
    args=p.parse_args()
    campaign=args.campaign or args.root/OUTPUT
    if args.output.resolve()==campaign.resolve() or campaign.resolve() in args.output.resolve().parents:
        raise ValueError('audit report must be outside immutable campaign evidence')
    if args.output.exists():raise FileExistsError('new audit output required; preserve earlier findings')
    receipt=None if args.external_output_readback is None else parse_json(read_regular(args.external_output_readback))
    result=audit_campaign(args.root,campaign,external_receipt=receipt)
    with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('assessment','runtime_status','runtime_reported_attempts','complete_verified_format_accepted_chunks','post_save_Git_evidence_verified')}))
    return 1 if result['assessment']=='evidence_UNKNOWN' else (2 if result['assessment']=='runtime_UNKNOWN' else 0)
if __name__=='__main__':raise SystemExit(main())
