"""Small glTF geometry adapter and explicit new reference-frame quality recipe."""
import numpy as np


def accessor(g,blob,index):
    a=g['accessors'][index];v=g['bufferViews'][a['bufferView']]
    if a.get('sparse') or a.get('normalized',False) or v['buffer']!=0:raise ValueError('No sparse/normalized/alternate buffer in this frozen scope')
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
    count=a['count'];stride=v.get('byteStride',width*dtype.itemsize);start=v.get('byteOffset',0)+a.get('byteOffset',0)
    if type(count) is not int or not 0<count<=300000 or stride<width*dtype.itemsize or stride%dtype.itemsize or start%dtype.itemsize:raise ValueError('Accessor shape/alignment')
    end=start+(count-1)*stride+width*dtype.itemsize
    if start<0 or end>len(blob) or end>v.get('byteOffset',0)+v['byteLength']:raise ValueError('Accessor byte bounds')
    return np.ndarray((count,width),dtype=dtype,buffer=blob,offset=start,strides=(stride,dtype.itemsize)).copy()


def read_geometry(g,blob):
    if g['asset']['version']!='2.0' or len(g['buffers'])!=1 or g['buffers'][0]['byteLength']!=len(blob):raise ValueError('Frozen core buffer differs')
    if g.get('extensionsRequired') or g.get('skins') or g.get('animations') or len(g['nodes'])!=1 or len(g['meshes'])!=1:raise ValueError('Only the declared static one-node objects are admitted')
    if g['scenes'][g.get('scene',0)]['nodes']!=[0]:raise ValueError('Unexpected scene root')
    n=g['nodes'][0];p,=g['meshes'][n['mesh']]['primitives']
    if p.get('mode',4)!=4 or p.get('extensions') or 'matrix' in n:raise ValueError('Only original static TRS triangles admitted')
    positions=accessor(g,blob,p['attributes']['POSITION']);indices=accessor(g,blob,p['indices']).reshape(-1)
    if positions.dtype!=np.float32 or positions.shape[1]!=3 or indices.dtype.kind not in 'ui' or len(indices)%3:raise ValueError('Position/index schema')
    if not np.isfinite(positions).all() or len(positions)>50000 or len(indices)>300000 or indices.max()>=len(positions):raise ValueError('Geometry values/size/index')
    declaration=g['accessors'][p['attributes']['POSITION']]
    bbox_error=max(float(np.max(np.abs(positions.min(0)-declaration['min']))),float(np.max(np.abs(positions.max(0)-declaration['max']))))
    if bbox_error>1e-7:raise ValueError('Independent glTF declared POSITION bounds disagree with binary')
    q=np.asarray(n.get('rotation',[0,0,0,1]),float);scale=np.asarray(n.get('scale',[1,1,1]),float);translation=np.asarray(n.get('translation',[0,0,0]),float)
    if q.shape!=(4,) or scale.shape!=(3,) or translation.shape!=(3,) or not np.isfinite(np.r_[q,scale,translation]).all() or abs(q@q-1)>1e-6 or (scale<=0).any():raise ValueError('Node TRS unsupported/invalid')
    x,y,z,w=q
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    world=(positions.astype(float)*scale)@R.T+translation
    center=(world.max(0)+world.min(0))/2;extent=float(np.ptp(world,axis=0).max())
    if not extent>0:raise ValueError('Empty object extent')
    normalized=((world-center)/extent).astype(np.float32)
    return normalized,indices.reshape(-1,3),{'source_POSITION_bbox_error':bbox_error,'source_world_bbox_min_m':world.min(0).tolist(),
        'source_world_bbox_max_m':world.max(0).tolist(),'reference_center_m':center.tolist(),'reference_longest_extent_m':extent,
        'coordinate':'native glTF right-handed Y-up meters; apply node TRS, center by reference bounds, divide reference longest extent; no prediction-specific fit',
        'reference_normalized_bbox_min':normalized.min(0).tolist(),'reference_normalized_bbox_max':normalized.max(0).tolist()}


def sample_surface(vertices,faces,seed,count=2048):
    triangles=vertices[faces].astype(np.float64)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
    if not np.isfinite(area).all() or (area<=0).any():raise ValueError('Do not drop degenerate faces')
    rng=np.random.default_rng(seed);chosen=rng.choice(len(faces),count,p=area/area.sum())
    r=np.sqrt(rng.random(count));s=rng.random(count)
    barycentric=np.stack([1-r,r*(1-s),r*s],axis=-1)
    return np.einsum('ni,nij->nj',barycentric,triangles[chosen]).astype(np.float32)


def point_metrics(predicted,reference):
    p=np.asarray(predicted,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    if r.shape!=(2048,3) or not np.isfinite(r).all():raise ValueError('Full frozen2048 reference points required')
    if p.shape!=(2048,3) or not np.isfinite(p).all():return {'passed':False,'planned_reference_points':2048,'failed_prediction':True,'Chamfer_L1':None,'Fscore_at_1percent':0.0}
    def nearest(a,b):
        return np.concatenate([np.linalg.norm(a[start:start+128,None,:]-b[None,:,:],axis=-1).min(axis=1) for start in range(0,len(a),128)])
    a,b=nearest(p,r),nearest(r,p);precision=float((a<=.01).mean());recall=float((b<=.01).mean())
    return {'passed':True,'planned_reference_points':2048,'prediction_points':2048,'Chamfer_L1':float(.5*(a.mean()+b.mean())),
        'Fscore_at_1percent':2*precision*recall/(precision+recall) if precision+recall else 0.,'precision':precision,'recall':recall,
        'definition':'symmetric mean unsquared Euclidean NN distance; inclusive threshold.01 in fixed reference-longest-extent1 frame; no ICP/recentering predicted points'}
