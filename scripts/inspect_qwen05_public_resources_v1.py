#!/usr/bin/env python3
"""Public fixed-revision metadata/config inspection; never weights or tokenizer calls."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REV = '7ae557604adf67be50417f59c2c2f167def9a775'
OUT = ROOT / 'official-public-metadata'
OUT.mkdir(exist_ok=False)
BASE = 'https://huggingface.co/'
URLS = [('model-metadata.json', BASE+'api/models/'+MODEL+'/revision/'+REV+'?blobs=true')]
URLS += [(name, BASE+MODEL+'/resolve/'+REV+'/'+name) for name in ['config.json', 'generation_config.json', 'tokenizer_config.json']]
rows=[]
for name,url in URLS:
    started=time.monotonic()
    req=urllib.request.Request(url,headers={'Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:
        data=response.read(1024*1024+1)
        if len(data)>1024*1024: raise ValueError('metadata cap exceeded')
        status=response.status
    with (OUT/name).open('xb') as f: f.write(data)
    rows.append({'name':name,'url':url,'status':status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'elapsed_seconds':time.monotonic()-started})
manifest={'schema':'qwen05-public-metadata-observation-v1','model':MODEL,'revision':REV,'actual_http_requests':len(rows),'automatic_retries':0,'authentication_used':False,'weights_downloaded':False,'model_loads':0,'tokenizer_calls':0,'generation_calls':0,'files':rows}
(OUT/'observation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
