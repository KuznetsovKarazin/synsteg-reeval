"""Equal-length lexical and inferred-theme controls; fixed C=1, no tuning.
The theme assignments use cover text only and are not recorded generation
prompt IDs. Therefore this is an inferred-theme holdout, not proof of an
exact prompt-disjoint generation experiment.
"""
from pathlib import Path
import json,re,collections,time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from _paths import parse_paths
SRC, OUT, SOURCE_ARCHIVE = parse_paths('Equal-length and inferred-theme controls')
previous=json.loads((OUT/'results.json').read_text());assign=json.loads((OUT/'topic_assignment.json').read_text());themes={r['uuid']:r['inferred_topic'] for r in assign['rows']}
D={};runs=[];t0=time.time()
for name,folder in [('luna','gpt6_luna_missing'),('qwen','qwen_missing_b3')]:
 D[name]=[json.loads(p.read_text()) for p in sorted((SRC/'reports'/folder).glob('*.json'))]
def lex(s):return re.findall(r'[a-z0-9]+',s.lower())
def flatten(rows,ids,equal=False):
 xx=[]; yy=[];uids=[]
 for i in ids:
  r=rows[i];a=r['container'];b=r['encoded_message']
  if equal:
   # Equalize raw words first as requested, strip formatting, then recrop
   # if punctuation normalization expanded different token counts.
   n=min(len(a.split()),len(b.split()));a=lex(' '.join(a.split()[:n]));b=lex(' '.join(b.split()[:n]));n=min(len(a),len(b));a=' '.join(a[:n]);b=' '.join(b[:n]);assert len(a.split())==len(b.split())
  xx.extend([a,b]);yy.extend([0,1]);uids.extend([(r['uuid'],'cover'),(r['uuid'],'stego')])
 return xx,np.array(yy),uids

def calc(rows,tr,te,setting,equal=False,seed=None,heldout=None):
 X,y,_=flatten(rows,tr,equal);Z,z,uids=flatten(rows,te,equal)
 v=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True);Xt=v.fit_transform(X);Zt=v.transform(Z)
 m=LogisticRegression(C=1,max_iter=2000,solver='lbfgs',random_state=0).fit(Xt,y);p=m.predict_proba(Zt)[:,1];pred=(p>=.5).astype(int)
 r={'setting':setting,'seed':seed,'heldout_inferred_theme':heldout,'accuracy':float(accuracy_score(z,pred)),'AUC':float(roc_auc_score(z,p)),'n_train_pairs':len(tr),'n_test_pairs':len(te),'errors':int((pred!=z).sum()),'confusion_matrix_01':confusion_matrix(z,pred,labels=[0,1]).tolist(),'train_uuids':[rows[i]['uuid'] for i in tr],'test_uuids':[rows[i]['uuid'] for i in te], 'predictions':[{'uuid':u,'arm':a,'true':int(yy),'pred':int(pp),'p_stego':float(sc)} for (u,a),yy,pp,sc in zip(uids,z,pred,p)]}
 runs.append(r);return r
for name,rows in D.items():
 byid={r['uuid']:i for i,r in enumerate(rows)}
 for r in previous['runs']:
  if r['setting']==name+'_pair_disjoint':calc(rows,[byid[u] for u in r['train_uuids']],[byid[u] for u in r['test_uuids']],name+'_equal_pair_length_normalized',True,r['seed'])
 for theme in sorted(set(themes[r['uuid']] for r in rows)):
  tr=[i for i,r in enumerate(rows) if themes[r['uuid']]!=theme];te=[i for i,r in enumerate(rows) if themes[r['uuid']]==theme]
  assert set(themes[rows[i]['uuid']] for i in tr).isdisjoint(themes[rows[i]['uuid']] for i in te)
  calc(rows,tr,te,name+'_leave_one_inferred_theme_out',heldout=theme)
summary={}
for s in sorted(set(r['setting'] for r in runs)):
 rr=[r for r in runs if r['setting']==s]
 if s.endswith('_out'):
  pp=[p for r in rr for p in r['predictions']];assert len(set((p['uuid'],p['arm']) for p in pp))==len(pp)
  summary[s]={'n_heldout_themes':len(rr),'accuracy_all_unique_texts':float(np.mean([p['true']==p['pred'] for p in pp])),'n_unique_texts':len(pp),'n_unique_pairs':len(pp)//2,'accuracy_macro_theme':float(np.mean([r['accuracy'] for r in rr])),'AUC_mean_by_theme':float(np.mean([r['AUC'] for r in rr])),'accuracy_min_theme':min(r['accuracy'] for r in rr),'accuracy_max_theme':max(r['accuracy'] for r in rr),'errors':sum(r['errors'] for r in rr),'per_theme':[{k:r[k] for k in ['heldout_inferred_theme','accuracy','AUC','n_test_pairs','errors']} for r in rr]}
 else: summary[s]={'n_runs':len(rr),'accuracy_mean':float(np.mean([r['accuracy'] for r in rr])),'accuracy_sd_population':float(np.std([r['accuracy'] for r in rr])),'AUC_mean':float(np.mean([r['AUC'] for r in rr])),'accuracy_per_seed':[r['accuracy'] for r in rr]}
(OUT/'additional_controls_results.json').write_text(json.dumps({'config':{'C':1,'max_iter':2000,'tfidf':'word 1-2grams, sublinear tf, fit on train only','theme_assignment':'cover-only inferred groups; no recorded prompt IDs','equal_length':'truncate to min raw word count, lowercase/alphanumeric normalization, recrop both to same normalized token count','runtime_sec':time.time()-t0},'runs':runs},indent=2))
(OUT/'additional_controls_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
