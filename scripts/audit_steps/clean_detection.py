"""Independent text-only steganalysis, fixed protocol, no test-set tuning.
All cover/stego variants of a UUID share one split. No keys, IDs, filenames,
requested length, or stored metrics are predictor inputs. All TF-IDF and
length scaling parameters are fit on training texts only.
"""
from pathlib import Path
import json, re, hashlib, time, sys, platform, collections
import numpy as np
import scipy, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from _paths import parse_paths
SRC, OUT, SOURCE_ARCHIVE = parse_paths('Independent text-only steganalysis')
SEEDS=[0,1,2,3,4]; t0=time.time(); results=[]
config={'seeds':SEEDS,'primary_C':1,'sensitivity_C':10,'tfidf':{'ngram_range':[1,2],'min_df':1,'sublinear_tf':True,'lowercase':True},'pair_split':[.68,.12,.20], 'near_duplicate_thresholds':{'character_5gram_tfidf_cosine':.80,'word_5shingle_jaccard':.50},'label_permutations':50,'predictor_inputs':'container / encoded_message text only','versions':{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__,'scipy':scipy.__version__}}
D={}
for name,folder in [('luna','gpt6_luna_missing'),('qwen','qwen_missing_b3')]:
 D[name]=[]
 for path in sorted((SRC/'reports'/folder).glob('*.json')):
  r=json.loads(path.read_text()); D[name].append({'uuid':r['uuid'],'cover':r['container'],'stego':r['encoded_message'],'L':len(r['message']),'source':str(path.relative_to(SRC))})
 assert len(set(r['uuid'] for r in D[name]))==len(D[name])

def canonical(s):return ' '.join(re.findall(r'[a-z0-9]+',s.lower()))
def flatten(rows,idx,mode='raw'):
 texts=[];ys=[];ids=[]
 for i in idx:
  r=rows[int(i)]
  for arm,y in [('cover',0),('stego',1)]:
   s=r[arm]
   if mode=='lexical_normalized':s=canonical(s)
   if mode=='first_30_words':s=' '.join(canonical(s).split()[:30])
   texts.append(s);ys.append(y);ids.append((r['uuid'],arm))
 return texts,np.array(ys),ids

def metrics(y,p,score):
 return {'accuracy':float(accuracy_score(y,p)),'AUC':float(roc_auc_score(y,score)),'n_texts':len(y),'errors':int((y!=p).sum()),'confusion_matrix_01':confusion_matrix(y,p,labels=[0,1]).tolist()}

def fit_eval(rows,tr,va,te,setting,seed,mode='raw',C=1,testrows=None):
 if testrows is None:
  testrows=rows
  assert set(tr).isdisjoint(te) and set(tr).isdisjoint(va) and set(va).isdisjoint(te)
 X,y,train_ids=flatten(rows,tr,mode); Xte,yte,test_ids=flatten(testrows,te,mode)
 # Test is never used to fit vocabulary, IDF, scaling, or model parameters.
 vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)
 xt=vec.fit_transform(X); xe=vec.transform(Xte)
 model=LogisticRegression(C=C,max_iter=2000,solver='lbfgs',random_state=0).fit(xt,y)
 score=model.predict_proba(xe)[:,1]; pred=(score>=.5).astype(int)
 rr={'setting':setting,'seed':seed,'mode':mode,'C':C,**metrics(yte,pred,score),
 'train_uuids':[rows[i]['uuid'] for i in tr],'validation_uuids':[rows[i]['uuid'] for i in va],'test_uuids':[testrows[i]['uuid'] for i in te],
 'n_train_pairs':len(tr),'n_validation_pairs':len(va),'n_test_pairs':len(te),'vocabulary_size':len(vec.vocabulary_),
 'exact_normalized_train_test_text_overlap':len(set(map(canonical,X))&set(map(canonical,Xte))),
 'predictions':[{'uuid':u,'arm':a,'true':int(yt),'pred':int(pr),'p_stego':float(sc)} for (u,a),yt,pr,sc in zip(test_ids,yte,pred,score)]}
 co=model.coef_[0];names=vec.get_feature_names_out(); rr['top_cover_features']=[{'feature':names[i],'coef':float(co[i])} for i in np.argsort(co)[:15]];rr['top_stego_features']=[{'feature':names[i],'coef':float(co[i])} for i in np.argsort(co)[-15:][::-1]]
 results.append(rr);print(setting,seed,mode,'C',C,rr['accuracy'],rr['AUC'],flush=True)
 return rr,xt,xe,y,yte

def primary_split(rows,seed):
 ii=np.arange(len(rows)); tr,te=train_test_split(ii,test_size=.2,stratify=[r['L'] for r in rows],random_state=seed)
 tr,va=train_test_split(tr,test_size=.15,stratify=[rows[i]['L'] for i in tr],random_state=seed)
 return tr.tolist(),va.tolist(),te.tolist()

# Group construction uses covers only; these features are NOT supplied to classifier.
allrows=D['luna']+D['qwen'];can=[canonical(r['cover']) for r in allrows]
cv=TfidfVectorizer(analyzer='char',ngram_range=(5,5),sublinear_tf=True).fit_transform(can)
cs=(cv@cv.T).toarray(); shingles=[set(tuple(s.split()[i:i+5]) for i in range(max(1,len(s.split())-4))) for s in can]
parent=list(range(len(allrows)));edges=[]
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):parent[find(i)]=find(j)
for i in range(len(allrows)):
 for j in range(i):
  jj=len(shingles[i]&shingles[j])/max(1,len(shingles[i]|shingles[j]))
  if can[i]==can[j] or cs[i,j]>=.80 or jj>=.50:
   union(i,j);edges.append({'uuid_a':allrows[i]['uuid'],'uuid_b':allrows[j]['uuid'],'cosine_char5':float(cs[i,j]),'jaccard_word5':jj,'exact':can[i]==can[j]})
groups=[find(i) for i in range(len(allrows))]
inventory={'config':config,'n_pairs':{k:len(v) for k,v in D.items()},'near_duplicate_edges':edges,'components':dict(collections.Counter(map(str,groups))),'exact_duplicate_covers':len(can)-len(set(can)),'cover_hashes':{r['uuid']:hashlib.sha256(r['cover'].encode()).hexdigest() for r in allrows}}
(OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))

for name,rows in D.items():
 for seed in SEEDS:
  tr,va,te=primary_split(rows,seed)
  rr,xt,xe,y,yte=fit_eval(rows,tr,va,te,name+'_pair_disjoint',seed)
  if seed==0:
   rng=np.random.default_rng(20261022);prs=[]
   for perm in range(50):
    # Independently flip labels within pairs on both sides: balanced null.
    yp=y^np.repeat(rng.integers(0,2,len(tr)),2);yep=yte^np.repeat(rng.integers(0,2,len(te)),2)
    m=LogisticRegression(C=1,max_iter=2000,solver='lbfgs',random_state=0).fit(xt,yp)
    pp=m.predict_proba(xe)[:,1];prs.append(metrics(yep,(pp>=.5).astype(int),pp))
   results.append({'setting':name+'_pair_label_permutation','seed':seed,'n_permutations':50,'accuracy_mean':float(np.mean([p['accuracy'] for p in prs])),'accuracy_min':min(p['accuracy'] for p in prs),'accuracy_max':max(p['accuracy'] for p in prs),'AUC_mean':float(np.mean([p['AUC'] for p in prs])),'null_exceedance_p':float((1+sum(p['accuracy']>=rr['accuracy'] for p in prs))/51),'runs':prs})
  fit_eval(rows,tr,va,te,name+'_C10_sensitivity',seed,C=10)
  fit_eval(rows,tr,va,te,name+'_lexical_normalized',seed,mode='lexical_normalized')
  fit_eval(rows,tr,va,te,name+'_first30',seed,mode='first_30_words')
  X,y,_=flatten(rows,tr);Xte,yte,_=flatten(rows,te)
  f=lambda xx:np.log(np.array([[max(1,len(x.split()))] for x in xx]))
  ss=StandardScaler().fit(f(X)); m=LogisticRegression(C=1,max_iter=2000).fit(ss.transform(f(X)),y)
  pp=m.predict_proba(ss.transform(f(Xte)))[:,1]; results.append({'setting':name+'_length_only','seed':seed,**metrics(yte,(pp>=.5).astype(int),pp)})
 offset=0 if name=='luna' else len(D['luna']); gg=groups[offset:offset+len(rows)]
 for seed in SEEDS:
  trva,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(np.arange(len(rows)),groups=gg))
  tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=seed).split(trva,groups=np.array(gg)[trva]))
  tr=trva[tr].tolist();va=trva[va].tolist();te=te.tolist()
  assert set(np.array(gg)[tr]).isdisjoint(np.array(gg)[te])
  fit_eval(rows,tr,va,te,name+'_near_duplicate_group_disjoint',seed)

for source,target in [('luna','qwen'),('qwen','luna')]:
 rows=D[source]; targets=D[target];so=0 if source=='luna' else len(D['luna']);to=0 if target=='luna' else len(D['luna'])
 fit_eval(rows,list(range(len(rows))),[],list(range(len(targets))),source+'_to_'+target+'_all',0,testrows=targets)
 targetgroups=set(groups[to:to+len(targets)]);tr=[i for i in range(len(rows)) if groups[so+i] not in targetgroups]
 fit_eval(rows,tr,[],list(range(len(targets))),source+'_to_'+target+'_near_duplicate_purged',0,testrows=targets)

summary={}
for s in sorted(set(r['setting'] for r in results)):
 rr=[r for r in results if r['setting']==s]
 if 'accuracy' in rr[0]: summary[s]={'runs':len(rr),'accuracy_mean':float(np.mean([r['accuracy'] for r in rr])),'accuracy_sd_population':float(np.std([r['accuracy'] for r in rr])),'AUC_mean':float(np.mean([r['AUC'] for r in rr])),'test_pair_counts':[r.get('n_test_pairs',r['n_texts']//2) for r in rr],'accuracy_per_run':[r['accuracy'] for r in rr]}
 else:summary[s]=rr[0]
(OUT/'results.json').write_text(json.dumps({'config':config,'runtime_seconds':time.time()-t0,'runs':results},indent=2))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print('DONE',time.time()-t0)
