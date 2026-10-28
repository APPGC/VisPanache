"""Render frozen saved products with the exact original overlay function."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np,xarray as xr
from original_renderer import plot_opacity_overlay_frame

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tracking',type=Path,required=True);p.add_argument('--products',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--refined',action='store_true');p.add_argument('--ffmpeg',default='ffmpeg');a=p.parse_args()
 if a.output.exists():p.error('output must be new')
 key='opacity_refined_raw' if a.refined else 'opacity_baseline';paths=sorted(a.products.glob('frame_*.nc'))
 if not paths:p.error('no products found')
 a.output.mkdir(parents=True)
 with xr.open_dataset(a.tracking,engine='h5netcdf') as ds:
  ids=list(map(str,ds.time.values))
  for i,path in enumerate(paths):
   index=ids.index(path.stem);img=ds.rgb.isel(time=index).values.astype('float32')/255
   with xr.open_dataset(path,engine='h5netcdf') as product:alpha=product[key].values.copy()
   plot_opacity_overlay_frame(img,alpha,out_png=a.output/(str(i).zfill(4)+'.png'),dpi=150)
 dest=a.output/('refined.mp4' if a.refined else 'baseline.mp4')
 subprocess.run([a.ffmpeg,'-nostdin','-v','error','-n','-framerate','10','-start_number','0','-i',str(a.output/'%04d.png'),'-frames:v',str(len(paths)),'-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=bicubic,setsar=1','-c:v','libx264','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
 (a.output/'render_manifest.json').write_text(json.dumps(dict(field=key,frame_ids=[p.stem for p in paths],preview_fps=10,original_renderer=True),indent=2)+'\n')
if __name__=='__main__':main()
