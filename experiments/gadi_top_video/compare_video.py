"""Compare new propagated masks with stored masks; agreement is not accuracy."""
from pathlib import Path
import json,csv
import numpy as np,xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segmentation_tools import mask_comparison,sky_weight

def compare_video(run,tracking,background,output):
 run=Path(run);output=Path(output);output.mkdir(mode=0o700,exist_ok=False)
 status=json.loads((run/'status.json').read_text())
 if status['status']!='PASS_MODEL_EXPORT_ONLY':raise RuntimeError('model run incomplete')
 records=json.loads((run/'stream_manifest.json').read_text());indexed={r['frame_index']:r for r in records}
 if set(indexed)!=set(range(22)):raise RuntimeError('incomplete manifest')
 w=sky_weight(np.load(background));upper=w>=.55;rows=[];previous=None
 with xr.open_dataset(tracking,engine='h5netcdf') as ds:
  if ds.sizes['time']!=22:raise ValueError('scope must remain 22 top frames')
  for i in range(22):
   rgb=ds.rgb.isel(time=i).values;baseline=ds.plume_mask.isel(time=i).values.astype(bool);frame=str(ds.time.values[i]);r=indexed[i]
   if r['frame_id']!=frame:raise ValueError('model/source frame mapping differs')
   with np.load(run/r['file']) as saved:new=saved['mask'].astype(bool)
   row=dict(frame_index=i,frame_id=frame,**mask_comparison(baseline,new,upper))
   row['successive_new_mask_iou']=mask_comparison(previous,new)['iou'] if previous is not None else None
   rows.append(row);previous=new.copy()
   fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
   for ax,title in zip(axes,['Stored independent mask','Point-guided video mask','Changes: orange added / blue removed']):ax.imshow(rgb);ax.axis('off');ax.set_title(title)
   axes[0].contour(baseline,levels=[.5],colors=['#cc33aa'],linewidths=.6);axes[1].contour(new,levels=[.5],colors=['#228833'],linewidths=.6)
   changed=np.full((*new.shape,4),0.,np.float32);changed[new&~baseline]=[1,.45,0,.7];changed[baseline&~new]=[0,.4,1,.7];axes[2].imshow(changed)
   fig.suptitle(frame+' | overlap is consistency, not smoke/cloud truth');fig.savefig(output/(frame+'.png'),dpi=130);plt.close(fig)
 with (output/'metrics.csv').open('w',newline='') as f:wcsv=csv.DictWriter(f,fieldnames=list(rows[0]));wcsv.writeheader();wcsv.writerows(rows)
 (output/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
 return rows
