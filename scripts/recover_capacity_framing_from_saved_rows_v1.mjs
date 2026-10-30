#!/usr/bin/env node
// public-dictionary-full-development-v3-transfer-recovery-v1
// No models, network, tokenizer, encoder or decoder.
// Run from repository root: node scripts/recover_capacity_framing_from_saved_rows_v1.mjs [output-directory]
import fs from 'node:fs';
import path from 'node:path';
const derive = function(dictionaryRows){
 const out=[];
 for(const r of dictionaryRows){
  for(const [framing,overhead] of [["plain_length16",16],["aes_siv",144]]){
   for(const kind of ["primary","secondary16"]){
    for(const parent of [128,256,512]){
     const bits=kind==="primary"?parent:16;
     const feasible=r.capacity_bits===null?null:r.capacity_bits>=bits+overhead;
     const x={capacity_bits:r.capacity_bits,capacity_feasible:feasible,delivery_status:"not_evaluated",dictionary_id:r.dictionary_id,feasibility_status:feasible===null?r.capacity_status:feasible?"mechanically_feasible":"mechanically_insufficient_capacity",frame_overhead_bits:overhead,framing,model_alias:r.model_alias,parent_primary_bits:parent,payload_bits:bits,payload_kind:kind,required_bits:bits+overhead,semantic_usability:"not_evaluated",source_order:r.source_order};
     out.push(x);
    }
   }
  }
 }
 return out;
};
const rowPaths = ["results/public-dictionary-full-development-v3/final/capacity-package/dictionary_rows-0000-0099.jsonl","results/public-dictionary-full-development-v3/final/capacity-package/dictionary_rows-0100-0199.jsonl","results/public-dictionary-full-development-v3/final/capacity-package/dictionary_rows-0200-0299.jsonl","results/public-dictionary-full-development-v3/final/capacity-package/dictionary_rows-0300-0399.jsonl"];
const originalFrames = ["results/public-dictionary-full-development-v3/final/capacity-package/frame_conditions-0000-0599.jsonl","results/public-dictionary-full-development-v3/final/capacity-package/frame_conditions-0600-1199.jsonl","results/public-dictionary-full-development-v3/final/capacity-package/frame_conditions-1200-1799.jsonl"];
const rows = rowPaths.flatMap(p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse));
if(rows.length!==400 || new Set(rows.map(r=>[r.model_alias,r.source_order,r.dictionary_id].join('/'))).size!==400) throw Error('invalid full grid');
for(const r of rows) if(r.capacity_bits!==null && (r.capacity_bits!==r.slots.reduce((a,s)=>a+s.width,0) || r.eligible_slots!==r.slots.length)) throw Error('slot sum mismatch');
const cells=derive(rows);
for(let n=0;n<3;n++) if(cells.slice(n*600,n*600+600).map(JSON.stringify).join('\n')+'\n'!==fs.readFileSync(originalFrames[n],'utf8')) throw Error('original framing byte mismatch');
const out=process.argv[2] || 'runtime/capacity-framing-transfer-recovery-v1';
fs.mkdirSync(out,{recursive:true});
for(let n=0;n<8;n++) fs.writeFileSync(path.join(out,'frame_conditions-'+String(n*600).padStart(4,'0')+'-'+String(n*600+599).padStart(4,'0')+'.jsonl'),cells.slice(n*600,n*600+600).map(JSON.stringify).join('\n')+'\n');
console.log(JSON.stringify({dictionary_rows:rows.length,frame_cells:cells.length,original_byte_matched_rows:1800,model_calls:0}));
