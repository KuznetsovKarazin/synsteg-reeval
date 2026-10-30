#!/usr/bin/env python3
"""Package/restore every public-v1 pilot primary file, with no neural execution."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("results/framed-interval-public-v1/native-pilot-v1")
ART=Path("results/framed-interval-public-v1-artifacts")
PART_LIMIT=512*1024
def sha(raw):return hashlib.sha256(raw).hexdigest()
def put(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n")
def safe(name):
 p=PurePosixPath(name)
 if p.is_absolute() or ".." in p.parts or str(p)!=name:raise ValueError("unsafe member")
 return p
def package(root=ROOT):
 root=Path(root);out=root/ART
 if out.exists():raise FileExistsError(out)
 source=root/SOURCE
 if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):raise ValueError("symlinkprimaryrefused")
 files={p.relative_to(root).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())}
  for p in sorted(source.rglob("*")) if p.is_file()}
 if not files:raise ValueError("no primary files")
 # One archive percase-endpoint; allremaining campaign/input/process evidence inmetadata.
 groups={}
 for name in files:
  parts=PurePosixPath(name).relative_to(SOURCE).parts
  group=parts[0]+"-"+parts[1] if len(parts)>=3 and parts[1] in {"sender","receiver"} else "metadata"
  groups.setdefault(group,[]).append(name)
 out.mkdir(parents=True)
 archives=[]
 for group,names in sorted(groups.items()):
  temp=out/(group+".zip.building")
  with zipfile.ZipFile(temp,"x",compression=zipfile.ZIP_LZMA) as z:
   for name in names:
    info=zipfile.ZipInfo(name,date_time=(2026,10,25,0,0,0));info.compress_type=zipfile.ZIP_LZMA;info.external_attr=0o100644<<16
    data=(root/name).read_bytes()
    if sha(data)!=files[name]["sha256"]:raise ValueError("primary changed during packaging")
    z.writestr(info,data)
  raw=temp.read_bytes();rows=[]
  chunks=[raw] if len(raw)<=1048576 else [raw[i:i+PART_LIMIT] for i in range(0,len(raw),PART_LIMIT)]
  for i,chunk in enumerate(chunks):
   path=out/(group+(".zip" if len(chunks)==1 else f".zip.part{i:03d}"))
   with path.open("xb") as f:f.write(chunk)
   rows.append({"path":path.relative_to(root).as_posix(),"bytes":len(chunk),"sha256":sha(chunk)})
  archives.append({"name":group,"zip_bytes":len(raw),"zip_sha256":sha(raw),"parts":rows,"members":names})
  temp.unlink()
 index={"schema":"framed-interval-public-v1-artifact-index","source":str(SOURCE),"files":files,
  "primary_files":len(files),"primary_bytes":sum(x["bytes"] for x in files.values()),"archives":archives,
  "archive_count":len(archives),"part_count":sum(len(a["parts"]) for a in archives),
  "native_calls":0,"includes_failures_unknowns_partial_writing_files":True}
 readable=[]
 for name,row in files.items():
  if name.endswith("/sender/committed-wire.bin"):
   data=(root/name).read_bytes()
   try:data.decode("utf-8",errors="strict")
   except UnicodeDecodeError:continue
   case=PurePosixPath(name).parts[-3];p=out/"readable"/(case+".txt");p.parent.mkdir(exist_ok=True)
   with p.open("xb") as f:f.write(data)
   readable.append({"path":p.relative_to(root).as_posix(),"source":name,**row})
 index["readable_texts"]=readable
 put(out/"index.json",index)
 return index
def restore(destination,root=ROOT):
 root=Path(root);index=json.loads((root/ART/"index.json").read_bytes());dest=Path(destination)
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir(parents=True)
 seen=set()
 for archive in index["archives"]:
  raw=b""
  for row in archive["parts"]:
   data=(root/row["path"]).read_bytes()
   if len(data)!=row["bytes"] or sha(data)!=row["sha256"]:raise ValueError("archivepartdiffers")
   raw+=data
  if len(raw)!=archive["zip_bytes"] or sha(raw)!=archive["zip_sha256"]:raise ValueError("archive differs")
  import io
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   if sorted(z.namelist())!=sorted(archive["members"]) or len(z.namelist())!=len(set(z.namelist())):raise ValueError("member set differs")
   for name in z.namelist():
    safe(name)
    if name in seen:raise ValueError("duplicate acrossarchives")
    seen.add(name);data=z.read(name);expected=index["files"][name]
    if len(data)!=expected["bytes"] or sha(data)!=expected["sha256"]:raise ValueError("primary member differs")
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("xb") as f:f.write(data)
 if seen!=set(index["files"]):raise ValueError("full denominator differs")
 return {"schema":"framed-interval-public-v1-artifact-fresh-restore","status":"passed",
  "verified_files":len(seen),"archive_count":index["archive_count"],"native_calls":0}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["package","restore"]);p.add_argument("--destination",type=Path)
 a=p.parse_args()
 if a.mode=="package":
  r=package();print(json.dumps({k:r[k] for k in ("primary_files","primary_bytes","archive_count","part_count")}))
 else:
  if a.destination is None:p.error("--destination required")
  print(json.dumps(restore(a.destination)))
