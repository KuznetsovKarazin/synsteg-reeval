"""Recompute source-only core findings from pinned development inputs, no generation."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit.shared_dictionary import PublicDictionary, Protocol, analyze_text

def run():
    folder=ROOT/'results/development-feasibility-v2'
    def rows(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    source=rows(ROOT/'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl')
    saved=rows(folder/'source-records.jsonl')
    assert len(source)==len(saved)==100
    assert all(r['split']=='development' for r in source)
    assert [r['record_id'] for r in source]==[r['record_id'] for r in saved]
    assert len({r['source_group_id'] for r in source})==100
    protocol=Protocol(min_group_size=1,max_bits_per_slot=1,framing='plain_length16')
    dictionaries={k:PublicDictionary.from_dict(json.loads((ROOT/f'data/shared_dictionary_recovery_{v}.json').read_text())) for k,v in [('baseline_v1','v1'),('strict_v2','v2')]}
    thresholds={'plain_8':24,'plain_16':32,'plain_32':48,'aes_siv_8':152,'aes_siv_32':176,'aes_siv_128':272}
    capacities={k:[] for k in dictionaries}
    checked=0
    for parent,row in zip(source,saved):
        assert parent['excerpt']==row['excerpt']
        assert hashlib.sha256(row['excerpt'].encode()).hexdigest()==row['excerpt_sha256']==parent['excerpt_sha256']
        assert parent['source_group_id']==row['source_group_id']
        n=len(parent['excerpt'].split())
        assert row['excerpt_words']==n and row['requested_to_excerpt_word_ratio']==parent['target_words']/n
        for label,dictionary in dictionaries.items():
            a=analyze_text(parent['excerpt'],dictionary,protocol)
            stored=row['capacity'][label]
            assert stored['capacity_bits']==a.capacity_bits
            assert stored['slot_occurrences']==[{'surface':s.surface,'class_id':s.class_id,'width_bits':s.width,'start':s.start,'end':s.end} for s in a.slots]
            assert stored['capacity_eligibility']=={k:a.capacity_bits>=v for k,v in thresholds.items()}
            capacities[label].append(a.capacity_bits);checked+=1
    summary=json.loads((folder/'summary.json').read_text())
    compact={}
    for label,values in capacities.items():
        stored=summary['source_excerpts']['capacity'][label]
        assert stored['text_denominator']==100
        assert stored['zero_capacity_texts']==values.count(0)
        for key,value in [('mean',statistics.mean(values)),('max',max(values)),('sum',sum(values)),('median',statistics.median(values))]:
            assert stored['capacity_bits'][key]==value
        compact[label]={'n':100,'zero':values.count(0),'mean_bits':statistics.mean(values),'max_bits':max(values),'eligible_plain8':sum(v>=24 for v in values)}
    length={}
    for target in [250,500]:
        group=[r for r in source if r['target_words']==target]
        assert len(group)==50
        length[str(target)]={'n':50,'median_source_words':statistics.median(len(r['excerpt'].split()) for r in group),'median_target_source_ratio':statistics.median(target/len(r['excerpt'].split()) for r in group)}
    return {'schema':'synsteg-feasibility-checkpoint-verification-v1','status':'passed','source_records':100,'source_dictionary_comparisons':checked,'source_capacity':compact,'length':length,'test_data_accessed':False,'new_generations':0,'paid_calls':0,'limits':['Checks source identities, source lengths, all source slot records and source capacity summaries.','Reuses the committed codec; not an independent codec implementation.','Does not replay cached full-lead extraction or re-evaluate qualitative cover judgments.','Synthetic unit tests are separate from these 100-source measurements.']}

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
