from pathlib import Path
import json,sys
import numpy as np,xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import argparse
parser=argparse.ArgumentParser(description='Render paper figures from reproduced NetCDF and analysis outputs; no opacity processing.')
for key in ['data','bottom-products','top-original-products','top-guided-products','refined-products','checks','output']:parser.add_argument('--'+key,type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('output must be new')
DATA=args.data;OUT=args.output;OUT.mkdir(parents=True)

plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig);print(name,flush=True)
def panel(ax,arr,title,opacity=False,cmap=None,vmin=None,vmax=None):
 kw=dict(cmap='inferno_r',vmin=0,vmax=1) if opacity else dict(cmap=cmap,vmin=vmin,vmax=vmax)
 im=ax.imshow(arr,**kw);ax.set_title(title,loc='left');ax.axis('off');return im
with xr.open_dataset(DATA/'bottom/plume_tracking.nc') as d: bottom=d.sel(time='frame_000817').load()
with xr.open_dataset(DATA/'top/plume_tracking.nc') as d: top=d.sel(time='frame_00376').load()
fig,axs=plt.subplots(1,2,figsize=(11,4.1));panel(axs[0],bottom.rgb.values,'(a) Bottom camera • frame 817');panel(axs[1],top.rgb.values,'(b) Top camera • frame 376');save(fig,'figure1_camera_views')
fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
def box(x,y,w,h,text,c='#e9eff5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',fc=c,ec='#52667b',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color='#52667b'))
box(.2,4.9,2.4,.65,'Observed RGB frames');box(3.3,4.9,2.6,.65,'Saved plume masks\nSAM3 prompt guidance');box(6.6,4.9,3,.65,'Background reference\nand terrain–sky weights')
box(.4,3.35,4,.70,'Terrain brightening branch');box(5.6,3.35,4,.70,'Sky attenuation branch');arrow(1.4,4.85,2.4,4.1);arrow(4.5,4.85,2.4,4.1);arrow(4.5,4.85,7.6,4.1);arrow(8.1,4.85,7.6,4.1);arrow(7.6,4.85,3.9,4.1);arrow(2,4.85,6.1,4.1)
box(2.2,2.05,5.6,.75,'Local cross-calibration → hybrid blend\nSign fallback → original local seam repair');arrow(2.4,3.3,3.6,2.85);arrow(7.6,3.3,6.4,2.85)
box(.15,.55,4.2,.75,'Preserved baseline + screened opacity\nRegime • reason flags • heuristic score',c='#e2f0e7');arrow(4,2,2.3,1.35)
box(5.6,.55,4.2,.75,'Optional seam-edge refinement\nSeparate flagged companion fields',c='#fff0d9');arrow(6,2,7.7,1.35)
save(fig,'figure2_workflow')
with xr.open_dataset(args.bottom_products/'frame_000817.nc') as d:q=d.load()
bg=np.load(DATA/'bottom/background.npy');w=np.load(DATA/'bottom/sky_weight.npy');rgb=bottom.rgb.values;mask=bottom.plume_mask.values.astype(bool)
fig,axs=plt.subplots(2,3,figsize=(11,6));panel(axs[0,0],rgb,'(a) Observed RGB');panel(axs[0,1],bg,'(b) Smoke-free reference');panel(axs[0,2],mask,'(c) Saved plume mask',cmap='gray',vmin=0,vmax=1);panel(axs[1,0],w,'(d) Input sky weight',cmap='Blues',vmin=0,vmax=1);im=panel(axs[1,1],q.opacity_baseline.values,'(e) Baseline apparent opacity',opacity=True);panel(axs[1,2],rgb,'(f) Original opacity overlay');a=q.opacity_baseline.values;axs[1,2].imshow(np.nan_to_num(a),cmap='inferno_r',vmin=0,vmax=1,alpha=np.where(np.isfinite(a),a,0));fig.colorbar(im,ax=axs[:,1:].ravel().tolist(),shrink=.6,pad=.02,label='Apparent opacity (1)');save(fig,'figure3_bottom_example')
a=np.load(args.checks/'arrays/bottom_frame_000700.npz');m=a['plume_mask'].astype(bool);ys,xs=np.where(m);sl=(slice(max(0,ys.min()-20),min(m.shape[0],ys.max()+20)),slice(max(0,xs.min()-20),min(m.shape[1],xs.max()+20)))
fig,axs=plt.subplots(2,2,figsize=(10,6));panel(axs[0,0],a['opacity_no_cal_no_seam'][sl],'(a) No cross-calibration or seam repair',opacity=True);im=panel(axs[0,1],a['opacity_no_seam'][sl],'(b) Cross-calibration; no seam repair',opacity=True);di=a['opacity_no_seam']-a['opacity_no_cal_no_seam'];lim=max(.01,float(np.nanmax(np.abs(di))));im2=panel(axs[1,0],di[sl],'(c) Cross-calibration increment',cmap='RdBu_r',vmin=-lim,vmax=lim);im3=panel(axs[1,1],(a['opacity_baseline']-a['opacity_no_seam'])[sl],'(d) Original seam-repair increment',cmap='magma',vmin=0,vmax=.3);fig.colorbar(im,ax=axs[0,:].tolist(),shrink=.7,label='Apparent opacity (1)');fig.colorbar(im2,ax=axs[1,0],shrink=.7,label='Change (1)');fig.colorbar(im3,ax=axs[1,1],shrink=.7,label='Change (1)');save(fig,'figure4_ablation')
with xr.open_dataset(args.refined_products/'frame_000700.nc') as product:
 b={'opacity_baseline':product.opacity_baseline.values.copy(),'opacity_refined':product.opacity_refined_raw.values.copy(),'modified_mask':product.seam_refinement_mask.values.copy()};changed=b['modified_mask'].astype(bool);ys,xs=np.where(changed);sl=(slice(max(0,ys.min()-25),min(changed.shape[0],ys.max()+26)),slice(max(0,xs.min()-25),min(changed.shape[1],xs.max()+26)))
fig,axs=plt.subplots(2,2,figsize=(10,5));panel(axs[0,0],a['opacity_no_seam'][sl],'(a) Seam repair off',opacity=True);panel(axs[0,1],a['opacity_baseline'][sl],'(b) Original seam repair',opacity=True);im=panel(axs[1,0],b['opacity_refined'][sl],'(c) Optional local edge refinement',opacity=True);im2=panel(axs[1,1],(b['opacity_refined']-b['opacity_baseline'])[sl],'(d) Optional increment only',cmap='magma',vmin=0,vmax=.3);fig.subplots_adjust(right=.80,wspace=.18,hspace=.35);fig.colorbar(im,cax=fig.add_axes([.82,.55,.018,.30]),label='Apparent opacity (1)');fig.colorbar(im2,cax=fig.add_axes([.82,.12,.018,.30]),label='Change (1)');save(fig,'figure5_seam_options')
rows=[]
for camera,folder in [('bottom',args.bottom_products),('top',args.top_original_products)]:
 for path in sorted(folder.glob('frame_*.nc')):
  with xr.open_dataset(path) as product:
   mask=product.plume_mask.values.astype(bool);valid=product.valid_opacity_mask.values.astype(bool);base=product.opacity_baseline.values;screen=product.opacity.values
   rows.append(dict(camera=camera,frame_id=path.stem,baseline_mean=float(base[mask].mean()),qc_mean=float(screen[valid].mean()),valid_fraction=float(valid.sum()/mask.sum()),plume_pixels=int(mask.sum()),seam_repaired_pixels=int(((product.quality_flag.values&256)>0).sum())))
fig,axs=plt.subplots(3,2,figsize=(10,7),sharex='col')
for j,c in enumerate(['bottom','top']):
 rr=[r for r in rows if r['camera']==c];ids=[int(r['frame_id'].split('_')[-1]) for r in rr]
 axs[0,j].plot(ids,[r['baseline_mean'] for r in rr],label='Baseline / raw mask');axs[0,j].plot(ids,[r['qc_mean'] for r in rr],label='Screened / valid support');axs[0,j].set_title(c.capitalize()+' camera');axs[0,j].set_ylabel('Conditional mean opacity');axs[0,j].legend(fontsize=7)
 axs[1,j].plot(ids,[r['valid_fraction']*100 for r in rr],color='#356a55');axs[1,j].set_ylabel('Retained plume pixels (%)')
 axs[2,j].plot(ids,[100*r['seam_repaired_pixels']/r['plume_pixels'] for r in rr],color='#af603d');axs[2,j].set_ylabel('Original seam repair (%)');axs[2,j].set_xlabel('Frame identifier')
 for ax in axs[:,j]:ax.grid(alpha=.2)
fig.tight_layout();save(fig,'figure6_sequence_diagnostics')
# Draw measured algorithm responses from the portable analysis output.
metrics=json.loads((args.checks/'ablation_background_metrics.json').read_text());fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,camera in zip(axs,['bottom','top']):
 for row in metrics:
  if row['camera']==camera:
   vals={r['background_offset_encoded_levels']:r['mean_absolute_opacity_change'] for r in row['background_sensitivity']};vals[0]=0.;xx=sorted(vals);ax.plot(xx,[vals[x] for x in xx],marker='o',label=row['frame_id'])
 ax.set_title(camera+' camera');ax.set_xlabel('Reference RGB offset (encoded 8-bit levels)');ax.set_ylabel('Mean absolute opacity change on raw mask');ax.legend(fontsize=8);ax.grid(alpha=.25)
fig.suptitle('Controlled reference mismatch; not measured reconstruction error');save(fig,'figure7_background_sensitivity')
fig,axs=plt.subplots(2,4,figsize=(12,5.2))
with xr.open_dataset(DATA/'top/plume_tracking.nc') as d:
 for row,f in enumerate(['frame_00367','frame_00376']):
  rgb=d.rgb.sel(time=f).values
  with xr.open_dataset(args.top_guided_products/(f+'.nc')) as p:
   mask=p.plume_mask.values if 'plume_mask' in p else np.isfinite(p.opacity_baseline.values)
   panel(axs[row,0],rgb,f'(a{row+1}) RGB {int(f.split("_")[-1])}');panel(axs[row,1],mask,f'(b{row+1}) Guided raw mask',cmap='gray',vmin=0,vmax=1);im=panel(axs[row,2],p.opacity_baseline.values,f'(c{row+1}) Baseline',opacity=True);panel(axs[row,3],p.opacity.values,f'(d{row+1}) Screened opacity',opacity=True)
fig.colorbar(im,ax=axs[:,2:].ravel().tolist(),shrink=.7,label='Apparent opacity (1)');save(fig,'figure8_top_failures')
thresholds=json.loads((args.checks/'threshold_rows.json').read_text());fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,camera in zip(axs,['bottom','top']):
 for near in [248,250,252]:
  fractions=[]
  for contrast in [1,3,5]:
   rr=[r for r in thresholds if r['camera']==camera and r['near_clipping_level']==near and r['contrast_levels']==contrast];fractions.append(sum(r['valid_pixels'] for r in rr)/sum(r['plume_pixels'] for r in rr))
  ax.plot([1,3,5],fractions,marker='o',label=f'Near clipping {near}/255')
 ax.set_title(camera.capitalize());ax.set_xlabel('Contrast tolerance (encoded levels /255)');ax.set_ylabel('Pixel-weighted retained fraction');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Threshold sensitivity • 5 bottom + 4 top frames');save(fig,'figureS1_qc_thresholds')
print('All nine paper figures rendered from reproduced products and metrics',flush=True)
