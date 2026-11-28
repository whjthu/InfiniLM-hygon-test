"""Map every foreground pixel in the frozen Poly Haven view to an original triangle."""
import argparse
import ast
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OLD=REPO/"local-only/sf3d-polyhaven-metal-tool-chest-fixed-view-v1/VIEW01"
VIEW=REPO/"tools/sf3d-polyhaven-metal-tool-chest-fixed-view-v1/view.py"
VIEW_SHA="658a9127286ff023fba570e41bca9f75966b0500f6495c2ac433392d41101536"
ARCHIVE=REPO/"results/bw1000/2026-11-27/specialized-sf3d-metal-tool-chest-fixed-view-cpu-pair.json"
ARCHIVE_SHA="f033b2798b07da8900b40fe75140f7a72d639a142d4a57c000a44d9aad0a3304"
COUNT=20600
FACE_COUNT=13360
SIZE=256
OUT="surface-pixel-lineage.private.npz"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def pin(raw):
    return {"bytes":len(raw),"SHA256":sha256(raw).hexdigest()}


def read(path,limit,expected):
    need(path.is_file() and not path.is_symlink(),"PINNED_INPUT_MUST_BE_REGULAR")
    with path.open("rb") as stream:
        raw=stream.read(limit+1)
    need(0<len(raw)<=limit and pin(raw)["SHA256"]==expected,"EXACT_FROZEN_INPUT_BYTES")
    return raw


def original():
    namespace={"__name__":"frozen_tool_chest_view","__file__":str(VIEW)}
    exec(compile(read(VIEW,32768,VIEW_SHA),str(VIEW),"exec"),namespace)
    freeze=json.loads(read(ARCHIVE,16384,ARCHIVE_SHA))
    fact=freeze["render"]
    frozen=json.loads(read(OLD/"RESULT.json",16384,freeze["private_render_SHA256"]))
    independent=json.loads(read(OLD/"VERIFY.json",8192,freeze["private_verify_SHA256"]))
    need(independent==freeze["independent"] and independent["passed"] is True
         and all(frozen[k]==v for k,v in fact.items()) and fact["pixel_targets"]==COUNT
         and fact["source_faces"]==fact["raster_faces"]==FACE_COUNT,"FROZEN_PRIOR_VIEW_AND_RECEIPTS")
    png=read(OLD/"view.private.png",512<<10,fact["RGBA"]["SHA256"])
    blob=read(OLD/"target.private.npz",4<<20,fact["PBR"]["SHA256"])
    need(pin(png)==fact["RGBA"] and pin(blob)==fact["PBR"],"EXACT_FROZEN_RGBA_AND_NPZ_EXTENTS")
    with Image.open(io.BytesIO(png),formats=("PNG",)) as img:
        need(img.mode=="RGBA" and img.size==(SIZE,SIZE),"FROZEN_RGBA_LAYOUT")
        rgba=np.asarray(img).copy()
    with np.load(io.BytesIO(blob),allow_pickle=False) as opened:
        need(set(opened.files)=={"mask","visible_uv","albedo","roughness","metallic"},"EXACT_PBR_INPUT_SCHEMA")
        targets={k:opened[k] for k in opened.files}
    return namespace,fact,rgba,targets


def capture_raster(refs):
    """Add only two outputs to the exact previous depth winner selection."""
    path=REPO/"tools/render_sf3d_boombox_material_views.py"
    body=read(path,32768,refs["FIXED"]).decode()
    f,=[node for node in ast.parse(body).body if isinstance(node,ast.FunctionDef) and node.name=="visible_uv"]
    code=ast.get_source_segment(body,f)
    def once(before,after):
        nonlocal code
        need(code.count(before)==1,"EXACT_REFERENCE_RASTER_ANCHOR")
        code=code.replace(before,after)
    once("    for face in faces:",
         "    face_id = np.full((size, size), -1, dtype=np.int32)\n"
         "    barycentric = np.zeros((size, size, 3), dtype=np.float64)\n"
         "    for face_index, face in enumerate(faces):")
    once("        depth[clip][replace] = z[replace]",
         "        depth[clip][replace] = z[replace]\n"
         "        face_id[clip][replace] = face_index\n"
         "        barycentric[clip][replace] = np.stack((wa[replace], wb[replace], wc[replace]), axis=1)")
    once("    return np.isfinite(depth), uv_map","    return np.isfinite(depth), uv_map, face_id, barycentric")
    scope={"np":np}
    exec(compile(code,"pinned original z-buffer with added lineage","exec"),scope)
    return scope["visible_uv"]


def geometry(namespace,reference,face_order,screen,uv):
    source=reference["source"]
    asset=namespace["ASSET"]
    obj=lambda key:read(asset/key,source[key]["bytes"],source[key]["SHA256"])
    gltf=json.loads(obj("metal_tool_chest.gltf"))
    binary=obj("metal_tool_chest.bin")
    mesh=namespace["module"]("code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py",namespace["MESH"])
    reader=mesh["accessor"]
    coords,world,indices,original,offsets,translations=[],[],[],[],[],[]
    offset=0
    faces=0
    for node_id,node in enumerate(gltf["nodes"]):
        mesh_id=node["mesh"]
        need(mesh_id==node_id and set(node)<={"name","mesh","translation"},"EXACT_SCENE_NODE_ORDER_AND_TRANSLATION")
        primitive,=gltf["meshes"][mesh_id]["primitives"]
        local=reader(gltf,binary,primitive["attributes"]["POSITION"]).astype(np.float64)
        index=reader(gltf,binary,primitive["indices"]).reshape(-1,3).astype(np.int32)
        translation=np.asarray(node["translation"],dtype=np.float64)
        original.append(index)
        coords.append(local)
        world.append(local+translation)
        indices.append(index+offset)
        translations.append(translation)
        offsets.append(faces)
        faces+=len(index)
        offset+=len(local)
    same=np.concatenate(indices)
    raw=np.concatenate(coords)
    w=np.concatenate(world)
    center=np.asarray(reference["center_m"],dtype=np.float64)
    extent=reference["extent_m"]
    norm=(w-center)/extent
    camera=np.asarray(reference["camera_direction"],dtype=np.float64)
    xz=np.cross([0.,1.,0.],camera)
    right=xz/np.linalg.norm(xz)
    up=np.cross(camera,right)
    matrix=np.stack([right,-up,camera]).T*(SIZE/reference["span"])
    offset_xyz=np.array([SIZE/2,SIZE/2,0.])
    need(len(offsets)==7 and faces==FACE_COUNT and offset==11073 and len(uv)==offset
         and np.array_equal(same,face_order)
         and np.max(np.abs(norm@matrix+offset_xyz-screen))<=3e-12,"ORIGINAL_WORLD_FACE_IDENTITY_AND_REPROJECTION")
    return raw,w,norm,np.concatenate(original),np.asarray(offsets),translations,matrix,offset_xyz


def verify_values(fields,mask,rgba,targets,screen,faces,uv,geom,base,arm,material):
    raw,world,normalized,original,offsets,translations,matrix,bias=geom
    yx=np.argwhere(mask)
    need(set(fields)=={"pixel_yx","source_global_face_id","source_mesh","source_primitive","source_face_id",
         "source_vertex_ids","barycentric","primitive_local_xyz_m","world_xyz_m","normalized_xyz","visible_uv","screen_depth"}
         and fields["pixel_yx"].dtype==np.uint16 and np.array_equal(fields["pixel_yx"],yx)
         and len(yx)==COUNT,"PIXEL_LINEAGE_SCHEMA_AND_DENOMINATOR")
    ids=fields["source_global_face_id"]
    rows=fields["source_mesh"]
    bary=fields["barycentric"]
    need(ids.dtype==np.int32 and ids.shape==(COUNT,) and ids.min()>=0 and ids.max()<FACE_COUNT
         and bary.shape==(COUNT,3) and np.isfinite(bary).all() and bary.min()>=-1e-12
         and np.max(np.abs(bary.sum(1)-1))<=1e-14,"SOURCE_FACE_AND_BARYCENTRIC_VALUES")
    mesh_ids=np.searchsorted(offsets,ids,side="right")-1
    need(np.array_equal(rows,mesh_ids) and np.array_equal(fields["source_face_id"],ids-offsets[mesh_ids])
         and not fields["source_primitive"].any() and np.array_equal(fields["source_vertex_ids"],original[ids]),
         "FACE_LOCAL_ORDER_MUST_MATCH_ORIGINAL_GLTF_INDEX_ACCESSOR")
    triangle=faces[ids]
    interp=lambda a:np.einsum("ni,nij->nj",bary,a[triangle])
    match=lambda a,b:np.max(np.abs(a-b))
    for name,vertices in (("primitive_local_xyz_m",raw),("world_xyz_m",world),("normalized_xyz",normalized),("visible_uv",uv)):
        need(np.isfinite(fields[name]).all() and match(fields[name],interp(vertices))<=1e-12,
             "INDEPENDENT_ORIGINAL_TRIANGLE_RECONSTRUCTION")
    local_world=fields["primitive_local_xyz_m"]+np.asarray(translations)[mesh_ids]
    expected=fields["normalized_xyz"]@matrix+bias
    simple=interp(screen)
    pixel_error=float(match(expected[:,:2],yx[:,::-1]+.5))
    uv_error=float(match(fields["visible_uv"],targets["visible_uv"]))
    need(pixel_error<=1e-10 and match(expected,simple)<=1e-10
         and match(simple[:,2],fields["screen_depth"])<=1e-10
         and match(local_world,fields["world_xyz_m"])<=1e-12 and uv_error<=1e-12,
         "REPROJECTED_PIXEL_CENTER_DEPTH_OR_ORIGINAL_UV_MISMATCH")
    lookup=np.floor((fields["visible_uv"]-np.floor(fields["visible_uv"]))*1024).astype(np.int64)
    rgb=base[lookup[:,1],lookup[:,0]]
    orm=arm[lookup[:,1],lookup[:,0]].astype(np.float64)/255.
    srgb=rgb.astype(np.float64)/255
    a=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    material_values=material["pbrMetallicRoughness"]
    need(material_values.get("baseColorFactor",[1,1,1,1])==[1,1,1,1]
         and material_values.get("roughnessFactor",1)==material_values.get("metallicFactor",1)==1,
         "EXACT_ORIGINAL_PBR_FACTORS")
    need(np.array_equal(mask,targets["mask"]) and np.array_equal(rgba[:,:,3],255*mask)
         and np.array_equal(rgb,rgba[mask,:3]) and not rgba[~mask].any()
         and np.array_equal(a,targets["albedo"]) and np.array_equal(orm[:,1],targets["roughness"])
         and np.array_equal(orm[:,2],targets["metallic"]),
         "BITWISE_RGBA_AND_THREE_PBR_FIELDS_REEXTRACT_MATCH")
    return dict(visible_pixels=COUNT,verified_RGBA_pixels=SIZE**2,matched_three_material_targets=COUNT,
                source_triangles=FACE_COUNT,source_meshes=7,visible_unique_faces=int(len(np.unique(ids))),
                visible_pixels_by_mesh=np.bincount(rows,minlength=7).tolist(),
                maximum_pixel_center_error_px=pixel_error,maximum_UV_error=uv_error,
                original_PBR_values_bitwise_equal=True)


def run(output,mode):
    output=output.resolve()
    need(output.is_relative_to(HERE) and output!=HERE,"PRIVATE_NEW_SCOPE_ONLY")
    namespace,fact,rgba,targets=original()
    refs,_,screen,faces,uv,base,arm,material,frame=namespace["native"]()
    for key in ("source","center_m","extent_m","camera_direction","span","source_faces","raster_faces"):
        need(fact[key]==frame[key],"EXACT_FROZEN_CAMERA_AND_WORLD_FRAME")
    geom=geometry(namespace,frame,faces,screen,uv)
    raw,world,normalized,original_indices,offsets,*_=geom
    mask,uv_map,face_id,barycentric=capture_raster(namespace)(screen,faces,uv,SIZE)
    need(np.array_equal(mask,targets["mask"]) and mask.sum()==COUNT,"ORIGINAL_VISIBILITY_SAME_Z_BUFFER")
    prior=targets["visible_uv"]
    need(np.array_equal(uv_map[mask],prior),"INDEXED_Z_BUFFER_PRESERVES_EXACT_FROZEN_UV")
    ids=face_id[mask]
    mesh=np.searchsorted(offsets,ids,side="right")-1
    bary=barycentric[mask]
    vertices=faces[ids]
    interp=lambda a:np.einsum("ni,nij->nj",bary,a[vertices])
    expected=dict(pixel_yx=np.argwhere(mask).astype(np.uint16),source_global_face_id=ids,source_mesh=mesh.astype(np.uint8),
        source_primitive=np.zeros(COUNT,dtype=np.uint8),source_face_id=(ids-offsets[mesh]).astype(np.int32),
        source_vertex_ids=original_indices[ids],barycentric=bary,primitive_local_xyz_m=interp(raw),world_xyz_m=interp(world),
        normalized_xyz=interp(normalized),visible_uv=uv_map[mask],screen_depth=interp(screen)[:,2])
    name=output/OUT
    if mode=="build":
        review=verify_values(expected,mask,rgba,targets,screen,faces,uv,geom,base,arm,material)
        output.mkdir(mode=0o700)
        with name.open("xb") as stream:
            np.savez_compressed(stream,**expected)
        manifest=dict(schema="SF3D_METAL_TOOL_CHEST_FIXED_VIEW_PIXEL_TRIANGLE_LINEAGE_V1",**review,
          source_asset="metal_tool_chest",source_license="CC0-1.0",source_payloads=frame["source"],
          original_view_public_SHA256=ARCHIVE_SHA,original_view_source_SHA256=VIEW_SHA,bundle=pin(name.read_bytes()),
          source_frame=dict(center_m=frame["center_m"],longest_extent_m=frame["extent_m"],
                            camera_to_surface_direction=list(-np.asarray(frame["camera_direction"])),
                            pixel_axes="y down, x right; pixel centers at integer + 0.5",orthographic_span=frame["span"],
                            forward_projection_row_matrix=geom[-2].tolist(),screen_offset=geom[-1].tolist(),
                            coordinates="glTF Y-up source meters; primitive local plus node translation gives world; (world-center)/longest extent"),
          indexing="zero-based original scene/mesh ordering; per-mesh primitive=0, source_face_id refers to original index accessor triangle",
          columns={k:{"shape":list(v.shape),"dtype":str(v.dtype),"raw_C_order_SHA256":sha256(v.tobytes()).hexdigest()}
                   for k,v in expected.items()},planned_examples=1,source_candidate_examples=1,actual_training_rows=0,
          native_features_created=0,actual_GPU=0,training_ready=False)
        with (output/"MANIFEST.json").open("x",encoding="utf8",newline="\n") as stream:
            json.dump(manifest,stream,sort_keys=True,allow_nan=False)
            stream.write("\n")
        return {"manifest":pin((output/"MANIFEST.json").read_bytes()),"bundle":manifest["bundle"],**review}
    manifest=json.loads((output/"MANIFEST.json").read_bytes())
    blob=read(name,12<<20,manifest["bundle"]["SHA256"])
    need(pin(blob)==manifest["bundle"] and manifest["source_payloads"]==frame["source"],"EXACT_LINEAGE_BUNDLE_AND_SOURCE_PIN")
    with np.load(io.BytesIO(blob),allow_pickle=False) as src:
        need(set(src.files)==set(expected),"EXACT_PERSISTED_LINEAGE_COLUMNS")
        observed={key:src[key] for key in src.files}
    for key,v in expected.items():
        got=observed[key]
        need(got.dtype==v.dtype and np.array_equal(got,v)
             and manifest["columns"][key]=={"shape":list(v.shape),"dtype":str(v.dtype),"raw_C_order_SHA256":sha256(got.tobytes()).hexdigest()},
             "EXACT_PERSISTED_PIXEL_TRIANGLE_PROVENANCE")
    review=verify_values(observed,mask,rgba,targets,screen,faces,uv,geom,base,arm,material)
    recorded=manifest["source_frame"]
    need(manifest["schema"]=="SF3D_METAL_TOOL_CHEST_FIXED_VIEW_PIXEL_TRIANGLE_LINEAGE_V1"
         and manifest["original_view_public_SHA256"]==ARCHIVE_SHA and manifest["original_view_source_SHA256"]==VIEW_SHA
         and manifest["source_asset"]=="metal_tool_chest" and manifest["source_license"]=="CC0-1.0"
         and recorded["center_m"]==frame["center_m"] and recorded["longest_extent_m"]==frame["extent_m"]
         and recorded["orthographic_span"]==frame["span"] and recorded["forward_projection_row_matrix"]==geom[-2].tolist()
         and recorded["screen_offset"]==geom[-1].tolist()
         and all(manifest[key]==value for key,value in review.items())
         and manifest["actual_training_rows"]==manifest["native_features_created"]==manifest["actual_GPU"]==0
         and manifest["training_ready"] is False,"INDEPENDENT_LINEAGE_MANIFEST_FRAME_AND_DENOMINATORS")
    return {"passed":True,"bundle":pin(blob),"manifest":pin((output/"MANIFEST.json").read_bytes()),**review}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("build","verify"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.mode),sort_keys=True,allow_nan=False))
