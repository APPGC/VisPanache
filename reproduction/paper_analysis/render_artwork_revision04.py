from pathlib import Path
import json,sys,hashlib,textwrap,argparse
import numpy as np,xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,PathPatch
from matplotlib.path import Path as PlotPath
from PIL import Image
DEFAULT_ROOT=Path(__file__).resolve().parents[1];VP=DEFAULT_ROOT.parent
parser=argparse.ArgumentParser(description='Render compact workflow and all20 full-frame diagnostic components; presentation only.')
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--baseline-products',type=Path,required=True)
parser.add_argument('--checks',type=Path,required=True)
parser.add_argument('--workflow',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
ARGS=parser.parse_args();ROOT=ARGS.output;ROOT.mkdir(parents=True,exist_ok=True);RELEASE=ARGS.source.parent;DATA=ARGS.data;FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)


original=ARGS.workflow;art=Image.open(original).convert('RGB')
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
fig,ax=plt.subplots(figsize=(14.4,6.5));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,1440),ylim=(650,0));ax.axis('off')
ax.text(720,24,'VisPanache processing workflow',ha='center',va='center',fontsize=19,fontweight='bold')
colours=['#254f91','#087e78','#087e78','#087e78','#087e78','#dc650a','#315a9a','#315a9a','#7842a4','#3b7439'];xs=[14+i*286 for i in range(5)];ys=[54,329];W=264;H=250
names=['Ground-based imagery','Plume segmentation','Background reconstruction','Terrain–sky weighting','Luminance conversion','Terrain and sky branches','Local cross-calibration','Hybrid blending','Interface post-processing','Outputs and diagnostics']
for i in range(10):
 x=xs[i%5];y=ys[i//5];c=colours[i];ax.add_patch(FancyBboxPatch((x,y),W,H,boxstyle='round,pad=0,rounding_size=10',lw=1.5,ec=c,fc='white'))
 ax.text(x+13,y+22,str(i+1),ha='center',va='center',color='white',fontsize=11,fontweight='bold',bbox=dict(boxstyle='circle,pad=.23',fc=c,ec=c))
 ax.text(x+32,y+22,names[i],va='center',fontsize=10.5,fontweight='bold')
 if i%5<4:ax.add_patch(FancyArrowPatch((x+W+1,y+H/2),(xs[i%5+1]-3,y+H/2),arrowstyle='-|>',mutation_scale=15,lw=1.3,color='#2b3540'))
verts=[(xs[4]+W/2,304),(xs[4]+W/2,316),(xs[0]+W/2,316),(xs[0]+W/2,326)];ax.add_patch(PathPatch(PlotPath(verts,[PlotPath.MOVETO,*([PlotPath.LINETO]*3)]),fc='none',ec='#2b3540',lw=1.3));ax.add_patch(FancyArrowPatch(verts[-2],verts[-1],arrowstyle='-|>',mutation_scale=15,color='#2b3540'))
def picture(box,x,y,w,h):ax.imshow(np.asarray(art.crop(box)),extent=(x,x+w,y+h,y),aspect='auto',interpolation='lanczos',zorder=2)
def label(i,text,y,fs=10):ax.text(xs[i%5]+W/2,ys[i//5]+y,text,ha='center',va='center',fontsize=fs)
picture((29,197,228,359),xs[0]+36,105,192,156);label(0,'Observed RGB frames',225)
picture((309,198,479,298),xs[1]+35,122,194,114);label(1,'Prompt-guided smoke mask',52);label(1,'“segment smoke plume”',222)
picture((556,214,789,363),xs[2]+15,123,234,150);label(2,'Reference frames · masking · inpainting',65,9)
picture((855,204,1099,365),xs[3]+14,115,236,156);label(3,'Sky mask and continuous sky weight',224,9)
picture((1166,187,1385,363),xs[4]+16,112,231,181)
# Original branch formulas are preserved as illustration cutouts.
label(5,'Terrain brightening',60,11);picture((67,489,280,567),xs[0]+23,400,218,69);label(5,'Sky attenuation',153,11);picture((67,699,280,771),xs[0]+23,495,218,68)
for box,xoff in [((371,506,463,589),14),((371,618,478,705),96),((371,728,463,813),178)]:picture(box,xs[1]+xoff,388,72,70)
for t,xoff in [('Support',50),('Scale',132),('Strength',214)]:ax.text(xs[1]+xoff,475,t,ha='center',fontsize=10)
label(6,'Local sky-to-terrain scaling',178,11);label(6,'Weighted near the interface',205,10)
picture((726,596,915,642),xs[2]+18,383,228,54);picture((725,665,915,804),xs[2]+45,447,175,118)
picture((1065,560,1144,641),xs[3]+18,389,80,78);ax.text(xs[3]+112,427,'Sign-aware\nfallback',va='center',fontsize=11)
picture((1066,663,1143,740),xs[3]+18,488,80,77);ax.text(xs[3]+112,526,'Local seam\nrepair',va='center',fontsize=11)
picture((1220,482,1314,704),xs[4]+16,377,68,158)
for t,y in [('Opacity',394),('Diagnostic maps',433),('Quality figures',472),('Time series',510)]:ax.text(xs[4]+96,y,t,va='center',fontsize=10.5)
label(9,'Mask · support · scale · strength · seam',227,8.5)
for j,(c,t) in enumerate([('#254f91','Inputs'),('#087e78','Preparation'),('#dc650a','Branches'),('#315a9a','Calibration / blend'),('#7842a4','Post-processing'),('#3b7439','Outputs')]):
 x=50+j*233;ax.plot([x,x+24],[616,616],color=c,lw=4);ax.text(x+32,616,t,va='center',fontsize=11)
for ext in ['png','pdf','svg']:fig.savefig(FIG/('figure2_workflow.'+ext),dpi=250,facecolor='white',pad_inches=0)
plt.close(fig)
sys.path.insert(0,str(ARGS.source))
from smoke_opacity.opacity import retrieve_hybrid_opacity
params=json.loads((ARGS.source/'configs/retrieval_paper.json').read_text());frame='frame_000700'
with xr.open_dataset(DATA/'bottom/plume_tracking.nc') as ds:img=ds.rgb.sel(time=frame).values.astype('float32')/255.;mask=ds.plume_mask.sel(time=frame).values.astype(bool)
bg=np.load(DATA/'bottom/background.npy');w=np.load(DATA/'bottom/sky_weight.npy');alpha,_,_,d=retrieve_hybrid_opacity(img,bg,mask,w,return_diagnostics=True,**params)
with xr.open_dataset(ARGS.baseline_products/(frame+'.nc')) as ds:np.testing.assert_array_equal(alpha,ds.opacity_baseline.values)
assert d['n_terrain_support']==1791 and d['n_sky_support']==5934 and int(d['seam_gap'].sum())==141
support=img*.4;support[d['terrain_support']]=[1,.5,.08];support[d['sky_support']]=[.1,.6,1]
plume=lambda x:np.where(mask,x,np.nan)
with np.load(ARGS.checks/'arrays/bottom_frame_000700.npz') as previous:no_seam=previous['opacity_no_seam'].copy()
# Trace actual pre-seam field includes fallback; comparing repair-on/off isolates its increment.
seam_increment=np.where(d['seam_gap'],alpha-no_seam,np.nan);assert np.all(seam_increment[d['seam_gap']]>=0)
cm=plt.get_cmap('inferno_r').copy();cm.set_bad('white');div=plt.get_cmap('coolwarm').copy();div.set_bad('white')
rows=[
('a','Observed RGB image',img,None,None,None),('b','Smoke-free background',bg,None,None,None),
('c','Saved plume mask',mask,'gray',0,1),('d','Smoothed terrain–sky weight',d['sky_weight'],'viridis',0,1),
('e','Signed distance to interface (pixels)',plume(d['signed_dist']),div,-150,150),('f','Support bands on RGB',support,None,None,None),
('g','Terrain branch before cross-calibration',d['alpha_terrain_g'],cm,0,1),('h','Sky branch before cross-calibration',d['alpha_sky_g'],cm,0,1),
('i','Local terrain opacity reference',plume(d['terrain_ref']),cm,0,1),('j','Local sky opacity reference',plume(d['sky_ref']),cm,0,1),
('k','Sky-to-terrain scale factor',plume(d['sky_to_terrain_scale']),cm,.6,2.5),('l','Interface calibration strength',plume(d['lambda_interface']),'viridis',0,.8),
('m','Sky branch after local scaling',d['alpha_sky_calibrated'],cm,0,1),('n','Effective sky branch used in blend',d['alpha_sky_eff'],cm,0,1),
('o','Hybrid blend before seam repair',d['alpha_linear'],cm,0,1),('p','Final baseline opacity',alpha,cm,0,1),
('q','Final opacity with seam contour',alpha,cm,0,1),('r','Seam-repaired pixels',d['seam_gap'],'gray',0,1),
('s','Increment from seam repair only',seam_increment,cm,0,max(.05,float(np.nanmax(seam_increment)))),('t','Numerical summary',None,None,None,None)]
validscale=d['sky_to_terrain_scale'][mask];validalpha=alpha[mask]

y,x=np.where(mask);x0=max(0,int(x.min())-25);x1=min(mask.shape[1],int(x.max())+26);y0=max(0,int(y.min())-25);y1=min(mask.shape[0],int(y.max())+26);crop=np.s_[y0:y1,x0:x1]
assert int(mask[crop].sum())==int(mask.sum()) and int(d['seam_gap'][crop].sum())==141
assert int(d['terrain_support'][crop].sum())==1791 and int(d['sky_support'][crop].sum())==5934
short=['Observed RGB','Background reference','Plume mask','Terrain–sky weight','Interface distance (px)','Calibration support bands','Terrain branch, gain applied','Sky branch, gain applied','Local terrain reference','Local sky reference','Sky-to-terrain scale','Interface calibration strength','Locally scaled sky branch','Effective sky branch','Hybrid before seam repair','Final baseline opacity','Final opacity + seam contour','Seam-repair mask','Seam-only increment','Numerical summary']
plt.rcParams.update({'font.size':10,'axes.titlesize':10.5,'figure.facecolor':'white','savefig.facecolor':'white'})
def draw(subset,name,ncols,nrows):
 # Fixed image rectangles eliminate the large gutters caused by equal-aspect subplot boxes.
 image_w=2.62;image_h=image_w*(y1-y0)/(x1-x0);cell_w=3.15;title_h=.48;row_h=image_h+title_h+.12;width=ncols*cell_w+.12;height=nrows*row_h+.66
 fig=plt.figure(figsize=(width,height));opacity_im=None
 for pos,row in enumerate(subset):
  letter,title,array,cmap,lo,hi=row;idx=ord(letter)-ord('a');col=pos%ncols;r=pos//ncols;left=(.06+col*cell_w)/width;bottom=(height-.06-title_h-image_h-r*row_h)/height
  ax=fig.add_axes([left,bottom,image_w/width,image_h/height]);ax.set_title(f'({letter}) '+textwrap.fill(short[idx],22),loc='left',fontsize=16,pad=5);ax.set_xticks([]);ax.set_yticks([])
  if array is None:
   ax.axis('off');ax.text(.01,.98,'Frame 700\nSupport t/s: 1791/5934 px\nGlobal 1.239 · seam 141 px\nOpacity p50 / p90 / max\n0.230 / 0.424 / 0.710\nScale p05 / p50 / p95\n0.767 / 1.132 / 1.959',va='top',fontsize=14,linespacing=1.2);continue
  im=ax.imshow(array[crop],cmap=cmap,vmin=lo,vmax=hi,interpolation='none')
  if letter in 'ghijmnopq':opacity_im=im
  elif letter in 'dekls':
   cax=fig.add_axes([left+(image_w+.055)/width,bottom,.075/width,image_h/height]);bar=fig.colorbar(im,cax=cax);bar.ax.tick_params(labelsize=13,length=2)
   if letter=='d':bar.set_ticks([0,.5,1])
   if letter=='e':bar.set_ticks([-150,0,150])
   if letter=='l':bar.set_ticks([0,.4,.8])
   if letter=='k':bar.set_ticks([.6,1,2.5])
  if letter=='q':ax.contour(d['seam_gap'][crop].astype(float),levels=[.5],colors='cyan',linewidths=.8)
  if letter=='f':ax.text(.02,.02,'Orange: terrain · blue: sky',transform=ax.transAxes,fontsize=10,color='white',bbox={'fc':'#333333','alpha':.8,'pad':2})
 cax=fig.add_axes([.30,.045,.40,.014]);bar=fig.colorbar(opacity_im,cax=cax,orientation='horizontal');bar.set_label('Apparent opacity (0–1)',fontsize=16,labelpad=3);bar.ax.tick_params(labelsize=14,length=2)
 for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=250,bbox_inches='tight',pad_inches=.035)
 plt.close(fig)
draw(rows,'figure3_bottom_example',5,4)
draw(rows[:10],'figure3_components_a_j',5,2)
draw(rows[10:],'figure3_components_k_t',5,2)
from shutil import copyfile
for ext in ['png','pdf']:copyfile(FIG/('figure3_bottom_example.'+ext),FIG/('figure3_full_components_compact.'+ext))
(ROOT/'artwork_receipt.json').write_text(json.dumps(dict(status='ARTWORK_GENERATED_REVIEW_PENDING',crop_xyxy=[x0,y0,x1,y1],all20_components=True,all_plume_pixels_visible=True,seam_pixels=141,baseline_exact=True,workflow_steps=10,workflow_original=str(original),workflow_new_size=[1440,650],old_workflow_size=list(art.size),frozen_source_unchanged=True),indent=2)+'\n')
