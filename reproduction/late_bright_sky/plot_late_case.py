from pathlib import Path
import json,hashlib,argparse
import numpy as np,xarray as xr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ROOT=a.bundle;OUT=ROOT/'expected';FIG=a.output;FIG.mkdir(exist_ok=False)
rows=json.loads((OUT/'metrics.json').read_text());items={};bounds=[]
for a in (2500,2600):
 with xr.open_dataset(OUT/'products'/f'frame_{a:06d}.nc') as ds:items[a]=ds.load()
 y,x=np.where(items[a].case_evaluated_mask.values.astype(bool));bounds.append((int(x.min()),int(x.max()+1)))
x0=max(0,min(b[0] for b in bounds)-20);x1=min(1364,max(b[1] for b in bounds)+20)
crop=np.s_[80:220,x0:x1];cmap=ListedColormap(['#dddddd','#d44236','#f6cf56','#6d3d99','#222222']);norm=BoundaryNorm(np.arange(-.5,5),5)
for a,ds in items.items():
 row=next(r for r in rows if r['frame_id']==f'frame_{a:06d}');image=np.asarray(Image.open(ROOT/'data/frames'/f'frame_{a:06d}.png').convert('RGB'));mask=ds.plume_mask.values.astype(bool);domain=ds.case_evaluated_mask.values.astype(bool);flags=ds.case_warning_flags.values
 assert not domain[:,:x0].any() and not domain[:,x1:].any()
 fig,axs=plt.subplots(3,2,figsize=(10.5,8.3),gridspec_kw={'height_ratios':[2.6,1,1]},layout='constrained');axs[0,0].imshow(image);axs[0,0].contour(mask,levels=[.5],colors=['#f29e00'],linewidths=.6)
 axs[0,0].add_patch(Rectangle((1080,80),200,140,fill=False,edgecolor='cyan',lw=1.3));axs[0,0].add_patch(Rectangle((x0,80),x1-x0,140,fill=False,edgecolor='white',lw=1,linestyle='--'));axs[0,0].set_title('(a) RGB, SAM3 boundary and diagnostic band');axs[0,1].imshow(mask,cmap='gray',vmin=0,vmax=1);axs[0,1].set_title('(b) Guided mask; boundaries unvalidated')
 for ax in axs[0]:ax.axis('off')
 for ax,name,title in [(axs[1,0],'archived_reference_contrast','(c) Observed − archived reference'),(axs[1,1],'same_row_control_contrast','(d) Observed − same-row sky control')]:
  im=ax.imshow(ds[name].values[crop],extent=(x0,x1,220,80),cmap='RdBu_r',vmin=-.2,vmax=.2);fig.colorbar(im,ax=ax,shrink=.8,fraction=.025,pad=.02,label='Encoded difference');ax.set_title(title)
 category=np.full(mask.shape,np.nan);category[domain]=0;category[(flags&1)!=0]=1;category[(flags&2)!=0]=2;category[(flags&8)!=0]=3;category[(flags&64)!=0]=4
 axs[2,0].imshow(category[crop],extent=(x0,x1,220,80),cmap=cmap,norm=norm);axs[2,0].set_title('(e) Red: bright; yellow: neutral; purple: opposing signs')
 endpoints=np.where(domain,((flags&48)!=0).astype(float),np.nan);axs[2,1].imshow(endpoints[crop],extent=(x0,x1,220,80),cmap='Reds',vmin=0,vmax=1);axs[2,1].contour(np.arange(x0,x1),np.arange(80,220),domain[crop],levels=[.5],colors=['#777'],linewidths=.5);axs[2,1].set_title(f'(f) No endpoint warnings; maximum {row["case_max_rgb"]}/255')
 for ax in axs[1:].ravel():ax.set_xlabel('Image x (pixel)',fontsize=8);ax.set_ylabel('Image y (pixel)',fontsize=8);ax.tick_params(labelsize=7)
 for ax in axs.ravel():ax.title.set_fontsize(9)
 fig.suptitle(f'Bottom frame {a}: bright plume against a changing sky\nSame-row control shift from archive: {row["control_shift_median"]:+.3f} encoded brightness',fontsize=12)
 fig.savefig(FIG/f'bright_sky_{a}.png',dpi=200);fig.savefig(FIG/f'bright_sky_{a}.pdf');plt.close(fig)
(FIG/'layout_receipt.json').write_text(json.dumps(dict(evaluated_rows=[80,220],crop_columns=[x0,x1],all_evaluated_anchor_pixels_visible=True,numerical_products_unchanged=True),indent=2)+'\n')
# Internal mask consistency is reported without treating overlap as reference accuracy.
summary=[]
for anchor in (2500,2600):
 subset=[r for r in rows if r['window']==anchor];areas=np.array([r['mask_pixels'] for r in subset]);main=next(r for r in subset if r['frame_id']==f'frame_{anchor:06d}');others=np.array([r['mask_pixels'] for r in subset if r is not main]);pairs=[];previous=None
 for r in subset:
  with xr.open_dataset(OUT/'products'/(r['frame_id']+'.nc')) as ds:m=ds.plume_mask.values.astype(bool)
  if previous is not None:pairs.append(float((m&previous).sum()/(m|previous).sum()))
  previous=m
 summary.append(dict(window=anchor,mask_area_min=int(areas.min()),mask_area_max=int(areas.max()),anchor_area=main['mask_pixels'],other_frame_area_median=float(np.median(others)),anchor_area_relative_to_other_median=float(main['mask_pixels']/np.median(others)-1),adjacent_iou_min=min(pairs),adjacent_iou_median=float(np.median(pairs)),interpretation='Conditional predicted-mask overlap with changing plume; not segmentation accuracy. Guided anchors show more ragged boundaries than propagated neighbours.'))
(FIG/'mask_consistency.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
