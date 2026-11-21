"""Explicit new glTF PBR atlas targets; not the original SF3D RGB heuristic."""
import numpy as np


def srgb_linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def decode_pixels(base_rgba,orm,material):
    if base_rgba.shape[:-1]!=orm.shape[:-1] or base_rgba.shape[-1]!=4 or orm.shape[-1]<3:
        raise ValueError('full matching source texture grids required')
    if base_rgba.dtype!=np.uint8 or orm.dtype!=np.uint8:raise ValueError('this recipe admits original8-bit textures only')
    if material.get('alphaMode','OPAQUE')!='OPAQUE':raise ValueError('alpha visibility needs a separate recipe')
    p=material['pbrMetallicRoughness'];factor=np.asarray(p.get('baseColorFactor',[1,1,1,1]),dtype=np.float64)
    if factor.shape!=(4,) or np.any(~np.isfinite(factor)) or np.any(factor<0) or np.any(factor>1):raise ValueError('baseColorFactor invalid')
    rough=float(p.get('roughnessFactor',1));metal=float(p.get('metallicFactor',1))
    if not np.isfinite([rough,metal]).all() or min(rough,metal)<0 or max(rough,metal)>1:raise ValueError('PBR factors invalid')
    return {'albedo':srgb_linear(base_rgba[...,:3].astype(np.float64)/255)*factor[:3],
        'roughness':orm[...,1].astype(np.float64)/255*rough,
        'metallic':orm[...,2].astype(np.float64)/255*metal}


def uniform_uv_sample(base_rgba,orm,material):
    height,width=base_rgba.shape[:2]
    if orm.shape[:2]!=(height,width):raise ValueError('texture grids differ')
    uv=(np.arange(16,dtype=np.float64)+.5)/16
    u,v=np.meshgrid(uv,uv,indexing='xy');xy=np.stack([np.floor(u*width),np.floor(v*height)],axis=-1).astype(np.int64)
    # glTF texture coordinates use the upper-left image origin. No resampling,
    # UV flip, geometry surface weighting, shader illumination or interpolation.
    pixels=decode_pixels(base_rgba[xy[...,1],xy[...,0]],orm[xy[...,1],xy[...,0]],material)
    return np.stack([u,v],axis=-1).reshape(256,2),{k:a.reshape((256,)+(a.shape[2:])) for k,a in pixels.items()}


def atlas_targets(base_rgba,orm,material):
    sums={'albedo':np.zeros(3,dtype=np.float64),'roughness':0.,'metallic':0.}
    count=0;extrema={k:[float('inf'),float('-inf')] for k in sums}
    for start in range(0,len(base_rgba),64):
        values=decode_pixels(base_rgba[start:start+64],orm[start:start+64],material)
        count+=values['roughness'].size
        for k,a in values.items():
            sums[k]+=a.sum(axis=(0,1));extrema[k][0]=min(extrema[k][0],float(a.min()));extrema[k][1]=max(extrema[k][1],float(a.max()))
    if count==0:raise ValueError('empty real texture')
    targets={k:np.asarray(v/count,dtype=np.float32).reshape(1,3) if k=='albedo' else np.asarray([v/count],dtype=np.float32) for k,v in sums.items()}
    if any(not np.isfinite(a).all() or np.any(a<0) or np.any(a>1) for a in targets.values()):raise ValueError('material target range')
    return targets,extrema,count


def compare_uv(prediction,reference):
    # This scorer is for future fixed UV probes, not camera-view LPIPS or the
    # original8 image reference. Missing/bad samples remain in the256 denominator.
    result={}
    for key,width in [('albedo',3),('roughness',1),('metallic',1)]:
        gold=np.asarray(reference[key],dtype=np.float64).reshape(256,width)
        pred=np.asarray(prediction[key],dtype=np.float64)
        if pred.shape!=(256,width):raise ValueError('all256 prediction slots required')
        valid=np.isfinite(pred).all(axis=1)&(pred>=0).all(axis=1)&(pred<=1).all(axis=1)
        # Range is [0,1]; failed samples incur full normalized unit error.
        per_sample=np.ones(256);per_sample[valid]=np.abs(pred[valid]-gold[valid]).mean(axis=1)
        result[key]={'planned':256,'valid':int(valid.sum()),'failed':int((~valid).sum()),'MAE_full_denominator':float(per_sample.mean())}
    return result
