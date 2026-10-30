#!/usr/bin/env python3
"""Exact transport chunks for existing public-v1 ZIPs; never regenerate/repack NN.

Git stores <=192KiB chunks. Logical ZIP identities and the original primary
index are unchanged. Restore rebuilds ZIPs in a temporary directory, then runs
the existing exact140-file restorer into a fresh user-selected destination.
"""
import argparse,hashlib,importlib.util,json,tempfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
ART=Path("results/framed-interval-public-v1-artifacts")
CHUNKS=ART/"transport-chunks-v1"
MANIFEST=CHUNKS/"manifest.json"
LIMIT=192*1024
HELPER="scripts/package_framed_interval_public_v1_artifacts.py"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def safe(value):
 p=PurePosixPath(value)
 if type(value) is not str or not value or p.is_absolute() or ".." in p.parts or str(p)!=value or "\\" in value:
  raise ValueError("unsafe transport path")
 return p
def put(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n")
def build(root=ROOT):
 root=Path(root);directory=root/CHUNKS
 if directory.exists():raise FileExistsError(directory)
 indexpath=root/ART/"index.json";indexraw=indexpath.read_bytes();index=json.loads(indexraw)
 rows=[]
 for archive in index["archives"]:
  if len(archive["parts"])!=1:raise ValueError("this transport version expects7logical singlepartZIPs")
  record=archive["parts"][0];path=record["path"];safe(path)
  raw=(root/path).read_bytes()
  if len(raw)!=record["bytes"] or sha(raw)!=record["sha256"] or sha(raw)!=archive["zip_sha256"]:
   raise ValueError("canonical ZIP bytes changed")
  rows.append((path,raw,record))
 directory.mkdir(parents=True)
 archives=[]
 for path,raw,record in rows:
  chunks=[]
  for offset in range(0,len(raw),LIMIT):
   data=raw[offset:offset+LIMIT];part=CHUNKS/(Path(path).name+f".part{offset//LIMIT:03d}")
   with (root/part).open("xb") as f:f.write(data)
   chunks.append({"path":part.as_posix(),"offset":offset,"bytes":len(data),"sha256":sha(data)})
  archives.append({"logical_zip_path":path,"bytes":len(raw),"sha256":sha(raw),"chunks":chunks})
 manifest={"schema":"framed-interval-public-v1-git-transport-chunks-v1","chunk_bytes_max":LIMIT,
  "original_index":{"path":(ART/"index.json").as_posix(),"bytes":len(indexraw),"sha256":sha(indexraw)},
  "original_logical_zip_count":len(archives),"archives":archives,
  "chunk_count":sum(len(r["chunks"]) for r in archives),
  "primary_files_unchanged":index["primary_files"],"primary_bytes_unchanged":index["primary_bytes"],
  "underlying_restorer":{"path":HELPER,"sha256":sha((root/HELPER).read_bytes())},
  "logical_zip_bytes_not_stored_directly_in_git":True,"recompression":False,"neural_calls":0}
 put(root/MANIFEST,manifest)
 return manifest
def assemble(root,destination):
 root=Path(root);dest=Path(destination)
 if dest.exists():raise FileExistsError(dest)
 manifest=json.loads((root/MANIFEST).read_bytes())
 if (manifest.get("schema")!="framed-interval-public-v1-git-transport-chunks-v1"
  or manifest.get("chunk_bytes_max")!=LIMIT or manifest.get("original_logical_zip_count")!=7):
  raise ValueError("unsupported transport manifest")
 descriptor=manifest["original_index"];safe(descriptor["path"])
 raw=(root/descriptor["path"]).read_bytes()
 if len(raw)!=descriptor["bytes"] or sha(raw)!=descriptor["sha256"]:raise ValueError("originalindexdiffers")
 index=json.loads(raw);expected={p["path"]:p for a in index["archives"] for p in a["parts"]}
 if set(expected)!={x["logical_zip_path"] for x in manifest["archives"]} or len(expected)!=len(manifest["archives"]):
  raise ValueError("logicalarchive denominator differs")
 pending=[];seen=set();chunkcount=0
 for a in manifest["archives"]:
  name=a["logical_zip_path"];safe(name);record=expected[name]
  if a["bytes"]!=record["bytes"] or a["sha256"]!=record["sha256"]:raise ValueError("logicalarchiveidentitydiffers")
  combined=bytearray()
  for c in a["chunks"]:
   p=c["path"];safe(p)
   if not p.startswith(CHUNKS.as_posix()+"/") or p in seen:raise ValueError("chunk path outsideclosedset")
   seen.add(p);chunkcount+=1
   if type(c["bytes"]) is not int or not 0<c["bytes"]<=LIMIT or c["offset"]!=len(combined):
    raise ValueError("noncontiguous oroversizedchunk")
   block=(root/p).read_bytes()
   if len(block)!=c["bytes"] or sha(block)!=c["sha256"]:raise ValueError("transportchunkdiffers")
   combined.extend(block)
  if len(combined)!=a["bytes"] or sha(combined)!=a["sha256"]:raise ValueError("logicalZIPreassemblydiffers")
  pending.append((name,bytes(combined)))
 if chunkcount!=manifest["chunk_count"]:raise ValueError("chunkdenominatordiffers")
 dest.mkdir(parents=True)
 for name,data in [(descriptor["path"],raw),*pending]:
  p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open("xb") as f:f.write(data)
 return {"schema":"framed-interval-public-v1-transport-reassembly","status":"passed",
  "logical_zip_count":len(pending),"chunk_count":chunkcount,"all_original_zip_bytes_exact":True,
  "original_index_sha256":descriptor["sha256"],"neural_calls":0}
def restore(destination,root=ROOT):
 root=Path(root);manifest=json.loads((root/MANIFEST).read_bytes());desc=manifest["underlying_restorer"]
 if desc["path"]!=HELPER or sha((root/HELPER).read_bytes())!=desc["sha256"]:
  raise ValueError("underlying frozen artifacthelper differs")
 spec=importlib.util.spec_from_file_location("public_primary_restorer",root/HELPER)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 with tempfile.TemporaryDirectory(prefix="public-native-logical-zips-") as td:
  staging=Path(td)/"assembled"
  assembly=assemble(root,staging)
  restored=module.restore(Path(destination),root=staging)
 return {"schema":"framed-interval-public-v1-chunk-restore","status":"passed",
  "transport":assembly,"primary":restored,"neural_calls":0}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["build","restore"])
 p.add_argument("--destination",type=Path);p.add_argument("--receipt",type=Path)
 a=p.parse_args()
 if a.receipt and a.receipt.exists():p.error("receipt exists")
 if a.mode=="build":result=build()
 else:
  if a.destination is None:p.error("--destination required")
  result=restore(a.destination)
 if a.receipt:put(a.receipt,result)
 print(json.dumps({"status":result.get("status","built"),"chunk_count":result.get("chunk_count",result.get("transport",{}).get("chunk_count")),
  "primary_files":result.get("primary_files_unchanged",result.get("primary",{}).get("verified_files"))}))
