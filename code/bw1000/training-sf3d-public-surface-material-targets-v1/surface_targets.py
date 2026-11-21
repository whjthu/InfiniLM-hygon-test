"""Explicit surface-area/nearest-texel global material recipe; no feature synthesis."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np

RECIPE='sf3d-public-surface-area-nearest-PBR4096-v1'
N=4096


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def fixed(path,wanted):
    with Path(path).open('rb') as f:b=f.read(wanted['bytes']+1)
    need(pin(b)==wanted,'Exact public source identity differs: '+Path(path).name)
    return b


def source_module(name,path,wanted):
    fixed(path,wanted)
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def sample_geometry(vertices,faces,uv,seed,count=N):
    v=np.asarray(vertices,dtype=np.float64);t=np.asarray(faces);u=np.asarray(uv,dtype=np.float64)
    need(v.ndim==2 and v.shape[1]==3 and u.shape==(len(v),2) and t.ndim==2 and t.shape[1]==3
         and t.dtype.kind in 'ui' and len(t)>0 and t.max()<len(v) and np.isfinite(v).all() and np.isfinite(u).all(),'Full source POSITION/index/UV contract differs')
    a=v[t];areas=.5*np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1)
    need(np.isfinite(areas).all() and (areas>0).all(),'Degenerate faces must not be silently removed')
    rng=np.random.default_rng(seed);chosen=rng.choice(len(t),count,p=areas/areas.sum())
    r=np.sqrt(rng.random(count));s=rng.random(count);bary=np.stack([1-r,r*(1-s),r*s],axis=1)
    sampled=np.einsum('ni,nij->nj',bary,u[t[chosen]])
    need(sampled.shape==(count,2) and np.isfinite(sampled).all(),'Surface UV denominator differs')
    return sampled,chosen.astype(np.int32),bary,{'original_triangles':len(t),'sampled_slots':count,
         'unique_sampled_triangles':len(np.unique(chosen)),'total_local_surface_area':float(areas.sum()),
         'minimum_triangle_area':float(areas.min()),'maximum_triangle_area':float(areas.max())}


def surface_samples(g,blob,accessor,base_rgba,orm,decode_pixels,group):
    p,=g['meshes'][0]['primitives'];material=g['materials'][p['material']]
    need(p.get('mode',4)==4 and not p.get('extensions') and material.get('alphaMode','OPAQUE')=='OPAQUE',
         'Only declared opaque static triangles admitted')
    node,=g['nodes'];scale=np.asarray(node.get('scale',[1,1,1]),dtype=float)
    need(not g.get('skins') and not g.get('animations') and 'matrix' not in node and scale.shape==(3,)
         and np.isfinite(scale).all() and (scale>0).all() and np.all(scale==scale[0]),
         'Area ratios require frozen static uniform-scale TRS')
    pbr=material['pbrMetallicRoughness']
    for key in ['baseColorTexture','metallicRoughnessTexture']:
        ref=pbr[key];need(ref.get('texCoord',0)==0 and not ref.get('extensions'),'Only original TEXCOORD_0 without transform admitted')
        tex=g['textures'][ref['index']];sampler=g.get('samplers',[])[tex['sampler']] if 'sampler' in tex else {}
        need(sampler.get('wrapS',10497)==sampler.get('wrapT',10497)==10497,'Frozen recipe admits repeat wrap only')
    v=accessor(g,blob,p['attributes']['POSITION']);f=accessor(g,blob,p['indices']).reshape(-1,3)
    uv=accessor(g,blob,p['attributes']['TEXCOORD_0'])
    seed=int(hashlib.sha256((group+'-surface-PBR4096-v1').encode()).hexdigest()[:16],16)
    sampled,chosen,bary,stats=sample_geometry(v,f,uv,seed)
    need(base_rgba.shape[:2]==orm.shape[:2] and base_rgba.dtype==orm.dtype==np.uint8,'Original texture grids/bitdepth differ')
    h,w=base_rgba.shape[:2]
    wrapped=np.mod(sampled,1.);xy=np.floor(wrapped*np.array([w,h])).astype(np.int64)
    values=decode_pixels(base_rgba[xy[:,1],xy[:,0]],orm[xy[:,1],xy[:,0]],material)
    result={'UV_surface':sampled,'triangle_index':chosen,'barycentric':bary.astype(np.float32)}
    for key,width in [('albedo',3),('roughness',1),('metallic',1)]:
        a=np.asarray(values[key]);need(a.shape==((N,3) if width==3 else (N,)) and np.isfinite(a).all()
             and (a>=0).all() and (a<=1).all(),'All real material sample slots required')
        result['sample_'+key]=a.astype(np.float32)
        result['native_batch_'+key]=np.asarray(a.mean(axis=0),dtype=np.float32).reshape((1,3) if width==3 else (1,))
    return result,dict(stats,seed=seed,nearest_base_level_texels=True,
        unspecified_original_texture_filter_is_not_shader_equivalence=True,
        normalized_coordinate_or_prediction_alignment=False)


def load_native_targets(path,wanted,recipe):
    need(recipe==RECIPE,'New surface recipe must be explicit; no original8 or old atlas inheritance')
    with np.load(io.BytesIO(fixed(path,wanted)),allow_pickle=False) as z:
        need(z['UV_surface'].shape==(N,2) and z['triangle_index'].shape==(N,) and z['barycentric'].shape==(N,3),
             'Complete surface sampling identity required')
        result={k:z['native_batch_'+k].copy() for k in ['albedo','roughness','metallic']}
        for k,a in result.items():
            need(a.shape==((1,3) if k=='albedo' else (1,)) and a.dtype==np.float32 and np.isfinite(a).all()
                 and (a>=0).all() and (a<=1).all(),'Frozen native target shapes/dtypes/range differ')
    return result


def training_identity(rows):
    selected=[r for r in rows if r['split']=='train' and r.get('completed')]
    need(len(selected)==2 and {r['group'] for r in selected}=={'WaterBottle','Corset'},
         'Both real train objects required; no development promotion or placeholder targets')
    doc={'recipe':RECIPE,'train_ordered':[{'group':r['group'],'source_mesh':r['mesh'],
          'source_material_textures':r['textures'],'target':r['target']} for r in selected]}
    return hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def to_original_target_object(fields):
    return {'albedo':fields['albedo'][0].copy(),'roughness':fields['roughness'].item(),
            'metallic':fields['metallic'].item()}
