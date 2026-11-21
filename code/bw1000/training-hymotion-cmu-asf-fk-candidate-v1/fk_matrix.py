"""Column-vector, right-handed ASF endpoint FK; no retargeting or learned model."""
import numpy as np


def axis_matrix(axis,angles):
    a=np.asarray(angles,dtype=np.float64);c=np.cos(np.deg2rad(a));s=np.sin(np.deg2rad(a))
    out=np.broadcast_to(np.eye(3),a.shape+(3,3)).copy()
    i,j={'x':(1,2),'y':(2,0),'z':(0,1)}[axis]
    out[...,i,i]=c;out[...,j,j]=c;out[...,i,j]=-s;out[...,j,i]=s
    return out


def ordered_rotation(order,angles):
    values=np.asarray(angles,dtype=np.float64)
    if values.shape[-1]!=len(order):raise ValueError('rotation values/order differ')
    out=np.broadcast_to(np.eye(3),values.shape[:-1]+(3,3)).copy()
    for i,axis in enumerate(order):out=axis_matrix(axis,values[...,i])@out
    return out


def solve(skeleton,motion):
    bones=skeleton['bones'];N=len(motion);names=['root']+[b['name'][0] for b in bones]
    byname={b['name'][0]:b for b in bones};parents={c:p for p,c in skeleton['hierarchy_edges']}
    offsets={};cursor=6
    for b in bones:
        n=len(b.get('dof',[]));offsets[b['name'][0]]=slice(cursor,cursor+n);cursor+=n
    if motion.shape!=(N,cursor) or not np.isfinite(motion).all():raise ValueError('complete native frame shape/finite required')
    scale=skeleton['root_length_to_meters']
    world={'root':ordered_rotation('xyz',motion[:,3:6])};ends={'root':motion[:,:3]*scale};bases={}
    def visit(name):
        if name in world:return
        b=byname[name];parent=parents[name];visit(parent)
        if b['axis'][3]!='XYZ':raise ValueError('this frozen recording requires XYZ axes')
        dof=b.get('dof',[])
        if len(set(dof))!=len(dof) or any(d not in ('rx','ry','rz') for d in dof):raise ValueError('rotation-only unique DOFs required')
        C=ordered_rotation('xyz',np.asarray(b['axis'][:3],dtype=np.float64))
        M=ordered_rotation(''.join(d[1] for d in dof),motion[:,offsets[name]])
        world[name]=world[parent]@C@M@C.T
        bases[name]=ends[parent]
        vector=np.asarray(b['direction'],dtype=np.float64)*float(b['length'][0])*scale
        ends[name]=bases[name]+np.einsum('nij,j->ni',world[name],vector)
    for name in names[1:]:visit(name)
    return dict(names=names,parents=[-1]+[names.index(parents[n]) for n in names[1:]],
        joint_endpoints_m=np.stack([ends[n] for n in names],axis=1),
        bone_bases_m=np.stack([bases[n] for n in names[1:]],axis=1),
        world_rotations=np.stack([world[n] for n in names],axis=1))
