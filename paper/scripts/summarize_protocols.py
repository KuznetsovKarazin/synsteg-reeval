import json,sys,statistics as st
from collections import defaultdict
for fn in sys.argv[1:]:
    d=json.load(open(fn)); print("==",fn, json.dumps(d['config'])[:600])
    g=defaultdict(list)
    for r in d['runs']:
        key=(r.get('setting',''),r['series'],r['detector'])
        g[key].append(r['test']['acc'])
    for k,v in g.items():
        print(k, len(v), round(st.mean(v),3), round(st.pstdev(v),3) if len(v)>1 else '')
