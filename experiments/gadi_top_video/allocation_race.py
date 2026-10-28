"""Atomic first-ready selection; only the winning PBS job may run model work."""
from pathlib import Path
import os,json,time,datetime,socket

def claim_allocation(root,job_id):
 root=Path(root);path=root/'allocation_winner.json'
 if not job_id:raise RuntimeError('PBS job identity required')
 record=dict(job_id=job_id,host=socket.gethostname(),claimed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='first allocation reaching atomic guard; only one model worker')
 try:
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 except FileExistsError:
  for _ in range(100):
   try:existing=json.loads(path.read_text());break
   except (json.JSONDecodeError,FileNotFoundError):time.sleep(.1)
  else:raise RuntimeError('Incomplete winner receipt; fail closed, preserve for diagnosis')
  return existing['job_id']==job_id,existing
 with os.fdopen(fd,'w') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return True,record
