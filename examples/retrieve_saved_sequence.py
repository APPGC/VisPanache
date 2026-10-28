"""Process saved RGB/masks using the normal package API; no model execution."""
from pathlib import Path
import argparse,json,hashlib,sys,platform,importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,xarray as xr
from smoke_opacity import retrieve_hybrid_opacity_qc,to_quality_dataset
from smoke_opacity.retrieval import retrieve,annotate_refinement

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['tracking','background','sky-weight','parameters','output','camera']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--inpaint-mask');p.add_argument('--mask-manifest',help='Optional SAM3 stream manifest; file entries relative to its directory')
    p.add_argument('--frames',nargs='+',help='Exact saved frame IDs, not physical timestamps')
    p.add_argument('--refine-seam-edges',action='store_true',help='Add flagged optional refinement companions; original opacity fields remain unchanged')
    args=p.parse_args();output=Path(args.output)
    if output.exists():p.error('output must be a new directory; existing products are never overwritten')
    params=json.loads(Path(args.parameters).read_text());bg=np.load(args.background);weight=np.load(args.sky_weight)
    inpaint=np.load(args.inpaint_mask).astype(bool) if args.inpaint_mask else None
    mask_index=None
    if args.mask_manifest:
        records=json.loads(Path(args.mask_manifest).read_text());mask_index={r['frame_id']:r for r in records}
        if len(mask_index)!=len(records):raise ValueError('duplicate frame IDs in mask manifest')
    paths=[args.tracking,args.background,args.sky_weight,args.parameters]+[v for v in [args.inpaint_mask,args.mask_manifest] if v]
    hashes={str(Path(path).resolve()):sha(path) for path in paths}
    environment={n:importlib.metadata.version(n) for n in ['numpy','scipy','xarray','h5netcdf','h5py']};environment.update(python=sys.version,platform=platform.platform())
    source=Path(__file__).resolve().parents[1]/'smoke_opacity';sources={p.name:sha(p) for p in source.glob('*.py')}
    with xr.open_dataset(args.tracking,engine='h5netcdf') as tracking:
        ids=list(map(str,tracking.time.values));selected=args.frames or ids
        if len(set(selected))!=len(selected) or not set(selected)<=set(ids):raise ValueError('unknown or duplicate requested frame IDs')
        if mask_index is not None and not set(selected)<=set(mask_index):raise ValueError('requested frame missing from mask manifest')
        output.mkdir(mode=0o700);completed=[]
        for frame in selected:
            i=ids.index(frame);img=tracking.rgb.isel(time=i).values;frame_hashes=dict(hashes)
            if mask_index is None:mask=tracking.plume_mask.isel(time=i).values.astype(bool)
            else:
                mp=Path(args.mask_manifest).parent/mask_index[frame]['file'];frame_hashes[str(mp.resolve())]=sha(mp)
                with np.load(mp) as a:mask=a['mask'].astype(bool)
            result=retrieve(img,bg,mask,weight,background_inpaint_mask=inpaint,refine_seam_edges=args.refine_seam_edges,**params)
            provenance=dict(camera_id=args.camera,frame_id=frame,source_version=sources,environment=environment,input_paths=list(frame_hashes),input_hashes=frame_hashes,time_interpretation='Frame IDs only; UTC/cadence not inferred')
            product=to_quality_dataset(result,provenance=provenance)
            product=annotate_refinement(product,result)
            dest=output/(frame+'.nc');product.to_netcdf(dest,engine='h5netcdf',encoding={n:dict(zlib=True,complevel=4) for n in product})
            with xr.open_dataset(dest,engine='h5netcdf') as saved:
                for name in product:np.testing.assert_array_equal(saved[name].values,product[name].values)
            completed.append(dict(frame_id=frame,file=dest.name,sha256=sha(dest),valid_pixels=int(result['valid_opacity_mask'].sum())))
            print(frame,'PASS full NetCDF roundtrip',flush=True)
        for path,h in hashes.items():
            if sha(path)!=h:raise RuntimeError('input changed: '+path)
        (output/'completion.json').write_text(json.dumps(dict(status='PASS_PACKAGE_REPLAY_AND_ARRAY_ROUNDTRIP',frames=completed,source_hashes=sources,input_hashes=hashes,environment=environment),indent=2)+'\n')
if __name__=='__main__':main()
