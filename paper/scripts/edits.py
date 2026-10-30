import json, glob, difflib, statistics as st
import os
REP=os.environ.get("SYNSTEG_REPORTS", "data/reports")
for s,d in {"luna":"gpt6_luna_missing","qwen3":"qwen_missing_b3","gpt4o":"gpt_omni_reports","ds2023":"json_reports"}.items():
    rates=[]; words=0; changed=0
    for fn in glob.glob(f"{REP}/{d}/*.json"):
        j=json.load(open(fn)); a=j["container"].split(); b=j["encoded_message"].split()
        sm=difflib.SequenceMatcher(a=a,b=b,autojunk=False)
        same=sum(m.size for m in sm.get_matching_blocks())
        rates.append(1-same/len(a)); words+=len(a); changed+=len(a)-same
        bits=len(j["message"])
    print(s, "mean edit", round(st.mean(rates),3), "pooled", round(changed/words,3))
