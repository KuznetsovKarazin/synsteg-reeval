#!/usr/bin/env python3
"""Aggregate actual saved generation metadata; no capacity analysis or inference."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

AUDIT_SHA='4d7cd207bdcb08d4e1f86488c79cdd9ac25f3e2c0189dc869d8d61000a1602f0'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--expected-manifest-sha256',required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();script=Path(__file__).with_name('audit_public_dictionary_full_development_v3.py')
    if hashlib.sha256(script.read_bytes()).hexdigest()!=AUDIT_SHA:raise ValueError('frozen audit source differs')
    spec=importlib.util.spec_from_file_location('capacity_metadata_binding_v3',script);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    raw=args.manifest.read_bytes();plan_raw=args.plan.read_bytes()
    a.require(a.sha(raw)==args.expected_manifest_sha256 and a.sha(plan_raw)==args.expected_plan_sha256,'external bindings differ')
    manifest=a.parse(raw);plan=a.parse(plan_raw);a.frozen_plan(plan)
    sources,grid,mapped,snapshot=a.validate_inputs(args.input_root,manifest,plan)
    rows=[];models={}
    for alias in plan['models']:
        tokens=[];reasons=[]
        for order in range(1,101):
            status=grid[alias,order]['status'];saved=mapped[alias,order]
            value=reason=hit=attempt_hash=None
            if saved is not None:
                data=(args.input_root/saved[1]).read_bytes();attempt=a.parse(data);result=attempt['result']
                value=result['usage']['output_tokens'];reason=result['stop_reason'];hit=result['hit_output_limit'];attempt_hash=a.sha(data)
                a.require(type(value)is int and 0<=value<=1024 and reason in ['end_turn','max_tokens'] and type(hit)is bool,'invalid saved token/stop metadata')
                tokens.append(value);reasons.append(reason)
            rows.append(dict(model_alias=alias,source_order=order,record_id=sources[order-1]['record_id'],cover_status=status,
                             output_tokens=value,stop_reason=reason,hit_output_limit=hit,attempt_sha256=attempt_hash))
        models[alias]=dict(full_source_denominator=100,observed_completed_attempts=len(tokens),
                           completed_output_tokens_sum=sum(tokens),stop_reason_counts=dict(Counter(reasons)),
                           unavailable_usage_rows=100-len(tokens),full100_output_token_total=None if len(tokens)<100 else sum(tokens))
    result=dict(schema='public-dictionary-full-development-v3-generation-usage-check',snapshot_commit=manifest['snapshot_commit'],
                input_manifest_sha256=a.sha(raw),plan_sha256=a.sha(plan_raw),script_sha256=a.sha(Path(__file__).read_bytes()),
                models=models,records=rows,model_calls=0,capacity_analysis_calls=0,encoder_calls=0,decoder_calls=0,
                scope='Sum of metadata from actual preserved completed attempts only; unknown/missing usage is null, not zero. No connection to the planned4800-neural-request experiment.')
    a.require(not args.output.exists() and not args.output.resolve().is_relative_to(args.input_root.resolve()),'output must be new and outside inputs')
    a.require(all(not p.is_symlink() for p in (args.output,*args.output.parents)),'linked output path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(models))


if __name__=='__main__':main()
