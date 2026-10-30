"""Static-protocol reproduction without importing/training neural models.
Only source texts, UUID, message length enter this diagnostic.
"""
from pathlib import Path
import json,re,collections,hashlib
import numpy as np
from sklearn.model_selection import train_test_split

from _paths import parse_paths
ROOT, OUT, SOURCE_ARCHIVE = parse_paths('Reproduce source document and sentence splits')
SENT=re.compile(r'(?<=[.!?])\s+')
norm=lambda t: ' '.join(t.casefold().split())

def loaded(series):
 return [json.loads(p.read_text()) for p in sorted((ROOT/'reports'/series).glob('*.json'))]

def overlap(X,ids,labels,tr,te):
 tx={X[i] for i in tr};tn={norm(X[i]) for i in tr};td={ids[i] for i in tr}
 train_labels=collections.defaultdict(set)
 for i in tr:train_labels[X[i]].add(int(labels[i]))
 return dict(n_train=len(tr),n_test=len(te),test_parent_in_train=sum(ids[i] in td for i in te),
  test_exact_text_in_train=sum(X[i] in tx for i in te),test_normalized_text_in_train=sum(norm(X[i]) in tn for i in te),
  test_text_opposite_label_in_train=sum((1-int(labels[i])) in train_labels[X[i]] for i in te),
  train_unique_texts_with_both_labels=sum(len(v)>1 for v in train_labels.values()))

source_archive=SOURCE_ARCHIVE
res={'source_archive_sha256':hashlib.sha256(source_archive.read_bytes()).hexdigest() if source_archive else None,
 'note':'Exact source split algorithms reproduced; no neural training. Counts are observations, not assumptions about semantic topic independence.','series':{}}
allreports={}
for series in ['gpt6_luna_missing','qwen_missing_b3']:
 reps=loaded(series);allreports[series]=reps
 docs={}
 for arm in ['container','encoded_message']:
  texts=[r[arm] for r in reps];freq=collections.Counter(texts)
  nf=collections.Counter(map(norm,texts));openings=collections.Counter(' '.join(t.split()[:8]).casefold() for t in texts)
  docs[arm]={'n':len(texts),'exact_unique':len(freq),'normalized_unique':len(nf),'duplicate_groups':sum(n>1 for n in freq.values()),
   'duplicate_records':sum(n for n in freq.values() if n>1),'opening8_unique':len(openings),'opening8_repeated_groups':sum(n>1 for n in openings.values()),'top_openings':openings.most_common(5)}
 docs['cover_stego_identical']=sum(r['container']==r['encoded_message'] for r in reps)
 docs['cover_stego_normalized_identical']=sum(norm(r['container'])==norm(r['encoded_message']) for r in reps)
 X=[t for r in reps for t in [r['container'],r['encoded_message']]];y=np.tile([0,1],len(reps));ids=[r['uuid'] for r in reps for _ in range(2)]
 ss=[]
 for seed in [0,1,2]:
  ix=np.arange(len(reps));L=[len(r['message']) for r in reps]
  tr,te=train_test_split(ix,test_size=.2,stratify=L,random_state=seed)
  tr,va=train_test_split(tr,test_size=.15,stratify=[L[i] for i in tr],random_state=seed)
  flat=lambda k: np.array([j for i in k for j in [2*i,2*i+1]])
  ss.append({'seed':seed,'train_test':overlap(X,ids,y,flat(tr),flat(te)), 'train_val':overlap(X,ids,y,flat(tr),flat(va)),
   'val_test':overlap(X,ids,y,flat(va),flat(te)), 'train_ids':[reps[i]['uuid'] for i in tr],'val_ids':[reps[i]['uuid'] for i in va],'test_ids':[reps[i]['uuid'] for i in te]})
 sp=[];spids=[];skipped=[]
 for r in reps:
  if not r.get('encoded_message'):continue
  c,s=SENT.split(r['container'].strip()),SENT.split(r['encoded_message'].strip())
  if len(c)!=len(s):skipped.append(r['uuid']);continue
  for a,b in zip(c,s):
   if a and b:sp.append((a,b));spids.append(r['uuid'])
 SX=[a for a,b in sp]+[b for a,b in sp];sy=np.r_[np.zeros(len(sp),int),np.ones(len(sp),int)];si=spids+spids
 sentence_seeds=[]
 for seed in [0,1,2]:
  tr,te=train_test_split(np.arange(len(SX)),test_size=.2,stratify=sy,random_state=seed)
  tr,va=train_test_split(tr,test_size=.15,stratify=sy[tr],random_state=seed)
  sentence_seeds.append({'seed':seed,'train_test':overlap(SX,si,sy,tr,te),'train_val':overlap(SX,si,sy,tr,va),'val_test':overlap(SX,si,sy,va,te)})
 res['series'][series]={'documents':docs,'whole_text_splits':ss,'sentence_pairs':len(sp),'identical_sentence_pairs':sum(a==b for a,b in sp),'sentence_mismatch_skipped_documents':skipped,'sentence_splits':sentence_seeds}
# Cross-generator whole-text exact overlap only (all possible original train subsets).
a,b=list(allreports.values());At={t for r in a for t in [r['container'],r['encoded_message']]};An={norm(t) for t in At}
Bt=[t for r in b for t in [r['container'],r['encoded_message']]]
res['cross_generator']={'luna_texts':len(a)*2,'qwen_texts':len(Bt),'qwen_exact_texts_in_luna':sum(t in At for t in Bt),'qwen_normalized_texts_in_luna':sum(norm(t) in An for t in Bt)}
(OUT/'split_diagnostics.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
short={s:{'documents':v['documents'],'sentence_pairs':v['sentence_pairs'],'identical_sentence_pairs':v['identical_sentence_pairs'],
 'sentence_skipped':len(v['sentence_mismatch_skipped_documents']),'whole_overlap':[{k:x[k] for k in ['seed','train_test']} for x in v['whole_text_splits']],
 'sentence_overlap':[{k:x[k] for k in ['seed','train_test']} for x in v['sentence_splits']]} for s,v in res['series'].items()}
short['cross_generator']=res['cross_generator']
(OUT/'split_summary.json').write_text(json.dumps(short,ensure_ascii=False,indent=2));print(json.dumps(short,ensure_ascii=False,indent=2))
