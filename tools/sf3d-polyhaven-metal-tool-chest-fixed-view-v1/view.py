"""Render and independently read one source-opaque, fixed-camera PBR reference."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ASSET=REPO/"local-only/polyhaven-metal-tool-chest-1k-domain-v1"
ADMISSION_SHA="91c6b769abcb038a5f3581bbb48675647eb57e02a66f139e60498ea2d17ab474"
ATLAS_SHA="f9d8af3d1a5c76096bcc26e30b11cb1d2bf7d001e2320b0168b188d8bb5f9f3b"
FIXED="9622d46acc67f34078a1b8bff88be89ae195af89392a111915d13aa365613415"
MESH="8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a"
PBR="3bc2279874fefbe2ebdb36ffa74824ee83661c63ea39359099dc266f7437d1e9"
SIZE=256
POSITION_COUNTS=[5910,1218,2476,576,1016,1588,576]


def need(ok,why):
    if not ok:
        raise ValueError(why)


def pin(raw):
    return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def read(path,cap,digest=None):
    need(path.is_file() and not path.is_symlink(),"EXACT_REGULAR_SCOPED_SOURCE")
    with path.open("rb") as f:
        raw=f.read(cap+1)
    need(0<len(raw)<=cap and (digest is None or sha256(raw).hexdigest()==digest),"EXACT_BOUNDED_ASSET_PIN")
    return raw


def module(path,digest):
    p=REPO/path
    ns={"__name__":"frozen_verified_source","__file__":str(p)}
    exec(compile(read(p,32768,digest),str(p),"exec"),ns)
    return ns


def decode(raw):
    with Image.open(io.BytesIO(raw),formats=("JPEG",)) as img:
        need(img.size==(1024,1024) and img.mode=="RGB","EXACT_PUBLISHER_1K_RGB_TEXTURE")
        return np.asarray(img).copy()


def native():
    refs=module("tools/render_sf3d_boombox_material_views.py",FIXED)
    mesh=module("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py",MESH)
    pbr=module("code/bw1000/training-sf3d-public-pbr-atlas2-v1/pbr.py",PBR)
    adm=json.loads(read(ASSET/"admission.json",16384,ADMISSION_SHA))
    atlas=json.loads(read(REPO/"results/bw1000/2026-11-27/specialized-sf3d-polyhaven-metal-tool-chest-native-atlas-target-readback.json",8192,ATLAS_SHA))
    need(adm["asset"]=="metal_tool_chest" and adm["license"]=="CC0-1.0"
         and atlas["independent"]["worker"]["admitted_member_SHA256"]==ADMISSION_SHA
         and atlas["independent"]["worker"]["alpha_mode"]=="OPAQUE","REUSE_EXACT_ADMITTED_OPAQUE_ASSET")
    expected=set(("metal_tool_chest.gltf","metal_tool_chest.bin",
                   *(f"textures/metal_tool_chest_{type}_1k.jpg" for type in ("diff","arm","nor_gl"))))
    need(set(adm["files"])==expected,"EXACT_FIVE_ADMITTED_SOURCE_FILES")
    data={}
    for name,record in adm["files"].items():
        raw=read(ASSET/name,record["bytes"],record["SHA256"])
        need(len(raw)==record["bytes"],"PUBLISHER_ASSET_EXTENT")
        data[name]=raw
    gltf=json.loads(data["metal_tool_chest.gltf"])
    geom=data["metal_tool_chest.bin"]
    need(gltf["asset"]["version"]=="2.0" and gltf["buffers"]==[{"uri":"metal_tool_chest.bin","byteLength":len(geom)}]
         and len(gltf["nodes"])==len(gltf["meshes"])==7 and gltf["scenes"][gltf["scene"]]["nodes"]==list(range(7))
         and not any(gltf.get(key) for key in ("skins","animations","extensionsRequired")),"EXACT_STATIC_TOOL_CHEST_SCENE")
    material,=gltf["materials"]
    need(material.get("alphaMode","OPAQUE")=="OPAQUE","OPAQUE_MATERIAL_ONLY")
    matrices=material["pbrMetallicRoughness"]
    names=[]
    for field in ("baseColorTexture","metallicRoughnessTexture"):
        binding=matrices[field]
        need(binding.get("texCoord",0)==0 and not binding.get("extensions"),"NO_TEXTURE_TRANSFORMS")
        names.append(gltf["images"][gltf["textures"][binding["index"]]["source"]]["uri"])
    need(names==["textures/metal_tool_chest_diff_1k.jpg","textures/metal_tool_chest_arm_1k.jpg"]
         and matrices.get("baseColorFactor",[1,1,1,1])==[1,1,1,1],"FROZEN_BASE_COLOR_AND_ROUGHNESS_METALLIC_SWIZZLE")
    verts,triangles,uvs=[],[],[]
    offset=0
    for index,node in enumerate(gltf["nodes"]):
        need(node["mesh"]==index and set(node)<={"name","mesh","translation"},"EXACT_NODE_TRANSLATION_ONLY")
        primitive,=gltf["meshes"][index]["primitives"]
        need(primitive.get("mode",4)==4 and primitive["material"]==0 and not primitive.get("extensions"),
             "EXACT_ORIGINAL_TRIANGLES")
        accessor=mesh["accessor"]
        positions=accessor(gltf,geom,primitive["attributes"]["POSITION"])
        uv=accessor(gltf,geom,primitive["attributes"]["TEXCOORD_0"])
        faces=accessor(gltf,geom,primitive["indices"]).reshape(-1,3)
        xyz=np.asarray(node["translation"],dtype=np.float64)
        need(positions.dtype==uv.dtype==np.float32 and positions.shape==(len(uv),3) and uv.shape[1]==2
             and xyz.shape==(3,) and np.isfinite(positions).all() and np.isfinite(uv).all() and np.isfinite(xyz).all()
             and faces.dtype.kind=="u" and faces.shape==(POSITION_COUNTS[index],3)
             and faces.max()<len(positions),"ORIGINAL_PRIMITIVE_GEOMETRY_AND_UV")
        verts.append(positions.astype(np.float64)+xyz)
        uvs.append(uv)
        triangles.append(faces.astype(np.int64)+offset)
        offset+=len(positions)
    world=np.concatenate(verts)
    uv=np.concatenate(uvs)
    all_faces=np.concatenate(triangles)
    center=(world.min(0)+world.max(0))/2
    extent=np.ptp(world,axis=0).max()
    need(extent>0 and len(world)==11073 and len(all_faces)==13360,"ALL_SEVEN_GEOMETRY_DENOMINATOR")
    camera=np.asarray(refs["CAMERAS"][0],dtype=np.float64)
    faces=all_faces
    if not material.get("doubleSided",False):
        a,b,c=world[faces].transpose(1,0,2)
        faces=faces[np.cross(b-a,c-a)@camera>0]
    screen=refs["project"]((world-center)/extent,camera,SIZE,refs["SPAN"])
    return refs,pbr,screen,faces,uv,decode(data[names[0]]),decode(data[names[1]]),material,dict(
        source={name:pin(value) for name,value in data.items()},center_m=center.tolist(),extent_m=float(extent),
        source_faces=len(all_faces),raster_faces=len(faces),camera_direction=camera.tolist(),span=refs["SPAN"])


def out(path):
    current=path.resolve()
    need(current.is_relative_to(HERE) and current!=HERE,"NEW_PRIVATE_OUTPUT_ONLY")
    return current


def render(folder):
    start=monotonic()
    refs,pbr,screen,faces,uv,base,orm,material,rec=native()
    mask,uv_map=refs["visible_uv"](screen,faces,uv,SIZE)
    need(0<mask.sum()<SIZE**2 and not np.any(mask[[0,-1],:]) and not np.any(mask[:,[0,-1]]),"NONEMPTY_UNCLIPPED_VISIBLE_PIXELS")
    selected=uv_map[mask]
    rgb=refs["sample_texture"](base,selected)
    orm=refs["sample_texture"](orm,selected)
    rgba=np.zeros((SIZE,SIZE,4),dtype=np.uint8)
    rgba[mask]=np.column_stack((rgb,np.full(len(rgb),255,dtype=np.uint8)))
    fields=pbr["decode_pixels"](rgba[mask],orm,material)
    folder=out(folder)
    folder.mkdir(mode=0o700)
    image=folder/"view.private.png"
    with image.open("xb") as stream:
        Image.fromarray(rgba).save(stream,format="PNG")
    blob=folder/"target.private.npz"
    with blob.open("xb") as stream:
        np.savez_compressed(stream,mask=mask,visible_uv=selected,**fields)
    report=dict(rec,asset="metal_tool_chest",renderer="fixed_camera_nearest_repeat_UV_unlit",planned_views=1,
       actual_views=1,height=SIZE,width=SIZE,pixel_targets=int(mask.sum()),opaque_foreground=True,lighting=False,
       native_training_rows=0,optimizer_updates=0,formal_delivery_increment=0,split="unassigned_source_candidate",
       RGBA=pin(image.read_bytes()),PBR=pin(blob.read_bytes()),wall_seconds=monotonic()-start)
    with (folder/"RESULT.json").open("x",encoding="utf8") as stream:
        json.dump(report,stream,sort_keys=True,allow_nan=False)
    return report


def verify(folder):
    refs,pbr,screen,faces,uv,base,orm,material,expected=native()
    folder=out(folder)
    result=json.loads(read(folder/"RESULT.json",8192))
    need(all(result[key]==value for key,value in expected.items()),"EXACT_PINNED_SOURCE_FRAME")
    png=read(folder/"view.private.png",512<<10,result["RGBA"]["SHA256"])
    with Image.open(io.BytesIO(png),formats=("PNG",)) as image:
        need(image.mode=="RGBA" and image.size==(SIZE,SIZE),"EXACT_OUTPUT_RGBA")
        rgba=np.asarray(image).copy()
    raw=read(folder/"target.private.npz",8<<20,result["PBR"]["SHA256"])
    with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
        target={k:arrays[k] for k in arrays.files}
    mask=target["mask"]
    need(mask.dtype==np.bool_ and mask.shape==(SIZE,SIZE) and np.array_equal(rgba[:,:,3],255*mask)
         and bool((rgba[~mask]==0).all()) and int(mask.sum())==result["pixel_targets"],"EXACT_ALPHA_AND_PBR_PIXELS_SHARE_MASK")
    pixels=target["visible_uv"]
    need(pixels.shape==(int(mask.sum()),2) and np.isfinite(pixels).all(),"EXACT_FINITE_VISIBLE_UV")
    tex=np.floor((pixels-np.floor(pixels))*1024).astype(np.int64)
    rgb=base[tex[:,1],tex[:,0]]
    rm=orm[tex[:,1],tex[:,0]].astype(np.float64)/255.
    srgb=rgb.astype(np.float64)/255.
    linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    need(np.array_equal(rgb,rgba[mask,:3])
         and np.array_equal(target["albedo"],linear)
         and np.array_equal(target["roughness"],rm[:,1]) and np.array_equal(target["metallic"],rm[:,2]),
         "INDEPENDENT_FULL_VISIBLE_COLOR_AND_PBR_RECOMPUTE")
    # Independent point-barycentric visibility test over a fixed set of foreground
    # and background pixels. The production raster works triangle-by-triangle.
    tri=screen[faces]
    a,b,c=tri.transpose(1,0,2)
    b,c=b-a,c-a
    determinant=b[:,0]*c[:,1]-b[:,1]*c[:,0]
    nz=np.abs(determinant)>=1e-10
    selected=np.full((SIZE,SIZE,2),np.nan)
    selected[mask]=pixels
    counts=[]
    for group in (np.argwhere(mask),np.argwhere(~mask)):
        for y,x in group[np.linspace(0,len(group)-1,24,dtype=int)]:
            rel=np.column_stack((x+.5-a[:,0],y+.5-a[:,1]))
            wb=np.divide(rel[:,0]*c[:,1]-rel[:,1]*c[:,0],determinant,where=nz,out=np.zeros(len(a)))
            wc=np.divide(b[:,0]*rel[:,1]-b[:,1]*rel[:,0],determinant,where=nz,out=np.zeros(len(a)))
            wa=1-wb-wc
            live=nz&(wa>=-1e-12)&(wb>=-1e-12)&(wc>=-1e-12)
            need(bool(live.any())==bool(mask[y,x]),"INDEPENDENT_VISIBLE_TRIANGLE")
            if live.any():
                z=np.where(live,wa*tri[:,0,2]+wb*tri[:,1,2]+wc*tri[:,2,2],-np.inf)
                i=int(z.argmax())
                need(np.allclose(np.array([wa[i],wb[i],wc[i]])@uv[faces[i]],selected[y,x],rtol=0,atol=1e-9),
                     "INDEPENDENT_NEAREST_SURFACE_UV")
        counts.append(24)
    return dict(passed=True,visible_pixels=int(mask.sum()),total_pixels=SIZE**2,
                independent_visibility_points=counts,actual_GPU=0,train_rows=0,rgba_pin=pin(png),pbr_pin=pin(raw),
                field_ranges={k:[float(target[k].min()),float(target[k].max())] for k in ("albedo","roughness","metallic")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("render","verify"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps((render if args.action=="render" else verify)(args.output),sort_keys=True,allow_nan=False))
