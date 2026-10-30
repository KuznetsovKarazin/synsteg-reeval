"""Cover-only inferred thematic groups; NOT recovered generator prompt IDs.
Read only uuid,container from source records. No stego texts or predictions used.
"""
from pathlib import Path
import json,re,collections
from _paths import parse_paths
ROOT, OUT, SOURCE_ARCHIVE = parse_paths('Cover-only inferred thematic groups')
RULES={
 'environment_energy':r'climate change|renewable energy|environmental conservation|sustainable living|conservation|greenhouse',
 'education':r'education|educational reform|learning',
 'social_media_friendship':r'social media|friendship|friendships',
 'ai_work_gig':r'artificial intelligence|\bai\b|automation|gig economy|future of work',
 'culture_arts_identity':r'globalization|cultural diversity|colonialism|indigenous|mass tourism|pop culture|cultural identity|\barts\b|\bart\b(?!\s+of\s+persuasion)|tourism',
 'health':r'older adults|healthcare|health care|mental health|healthy lifestyles|aging population|ageing population',
 'security_privacy':r'cyber threats|cybersecurity|privacy|big data',
 'bioethics_nanotechnology':r'animals in research|genetic engineering|cloning|reproductive technologies|animal testing|nanotechnology|nanoparticles',
 'urban_population':r'overpopulation|urbanization|urbanisation',
 'media_expression':r'media bias|propaganda|censorship|freedom of speech|persuasion|advertising',
 'economy_corporate_consumption':r'economic inequality|alternative economic|economic systems|corporate social responsibility|social responsibility of corporations|ethical consumption|corporations|responsible business',
 'family_happiness':r'\bfamily\b|happiness|well[ -]being|wellness',
 'space':r'space exploration|exploring space',
 'vr_games':r'virtual reality|video games|video gaming',
 'politics_justice_equality':r'women in leadership|women.*?leadership|racial and gender|gender equality|racial equality|youth participation|young people.*?politic|criminal justice|justice reform',
 'technology_general_divide':r'\btechnology\b|digital divide|technological',
}
rows=[]
for series in ['gpt6_luna_missing','qwen_missing_b3']:
 for f in sorted((ROOT/'reports'/series).glob('*.json')):
  r=json.loads(f.read_text()); opening=r['container'][:320].lower(); matches=[]
  for group,pat in RULES.items():
   m=re.search(pat,opening)
   if m:matches.append((m.start(),-len(m.group()),group,m.group()))
  matches.sort(); chosen=matches[0] if matches else None
  # AI in education is grouped with all education prompts; early cover context only.
  if chosen and chosen[2]=='ai_work_gig' and chosen[3] in ('artificial intelligence','ai') and re.search(r'student|teacher|lesson|tutor',opening[:220]):
   chosen=(chosen[0],chosen[1],'education','AI in education: early-context rule')
  rows.append({'uuid':r['uuid'],'series':series,'inferred_topic':chosen[2] if chosen else 'UNRESOLVED','anchor':chosen[3] if chosen else None,'anchor_position':chosen[0] if chosen else None,'opening':r['container'][:220],'candidate_groups':[x[2] for x in matches]})
meta={'method':'Fixed regular-expression anchors applied only to first320 characters of cover. Earliest anchor wins. AI early contexts containing student/teacher/lesson/tutor group with education. All openings are exported for inspection. These are inferred broad thematic groups, not logged prompt IDs.','original_topic_count':51,'prompt_uuid_mapping_available':False,'recorded_generation_seed_available':False,'counts':{s:dict(collections.Counter(r['inferred_topic'] for r in rows if r['series']==s)) for s in ['gpt6_luna_missing','qwen_missing_b3']},'rows':rows}
(OUT/'topic_assignment.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(json.dumps(meta['counts'],indent=2))
for r in rows:
 if r['inferred_topic']=='UNRESOLVED' or r['anchor_position']>60:
  print('CHECK',r['uuid'],r['inferred_topic'],r['anchor_position'],r['opening'])
