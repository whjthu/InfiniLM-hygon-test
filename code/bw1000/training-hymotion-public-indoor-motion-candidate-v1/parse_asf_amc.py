"""CMU-native parsing and root-unit bridge, not an HY-Motion201 retargeter."""
import math


def need(ok,why):
    if not ok:raise ValueError(why)


def numbers(parts):
    result=[float(x) for x in parts]
    need(all(math.isfinite(x) for x in result),'nonfinite motion/skeleton field')
    return result


def parse_asf(raw):
    text=raw.decode('ascii');section=None;bone=None;bones=[];units={};root={};edges=[]
    for original in text.splitlines():
        line=original.partition('#')[0].strip()
        if not line:continue
        if line.startswith(':'):
            section=line[1:].split()[0].lower();continue
        parts=line.split();key=parts[0].lower();values=parts[1:]
        if section=='units':units[key]=values
        elif section=='root':root[key]=values
        elif section=='bonedata':
            if key=='begin':need(bone is None,'nested ASF bone');bone={}
            elif key=='end':
                need(bone is not None and all(k in bone for k in ('id','name','direction','length','axis')),'incomplete ASF bone')
                bones.append(bone);bone=None
            elif bone is not None and key in ('id','name','direction','length','axis','dof'):bone[key]=values
        elif section=='hierarchy' and key not in ('begin','end'):
            edges.extend((parts[0],x) for x in values)
    need(bone is None and units.get('angle')==['deg'] and len(units.get('length',[]))==1,'ASF units or final block differs')
    scale=float(units['length'][0]);need(math.isfinite(scale) and scale>0,'ASF length scale invalid')
    need(root.get('order')==['TX','TY','TZ','RX','RY','RZ'],'native root channel order differs')
    named={b['name'][0]:b for b in bones};need(len(named)==len(bones)>0 and len({int(b['id'][0]) for b in bones})==len(bones),'ASF bone names/IDs repeat')
    for b in bones:
        need(len(numbers(b['direction']))==3 and len(numbers(b['length']))==1 and float(b['length'][0])>=0,'ASF direction/length invalid')
        need(len(b['axis'])==4 and len(numbers(b['axis'][:3]))==3 and b['axis'][3]=='XYZ','ASF bone axis differs')
        need(all(x in ('rx','ry','rz','tx','ty','tz') for x in b.get('dof',[])),'unknown native DOF')
    parents={}
    for parent,child in edges:
        need(parent=='root' or parent in named,'unknown hierarchy parent')
        need(child in named and child not in parents,'unknown or multiply-parented bone');parents[child]=parent
    need(set(parents)==set(named),'hierarchy does not cover all bones')
    for name in named:
        seen=set();current=name
        while current!='root':
            need(current not in seen,'cyclic ASF hierarchy');seen.add(current);current=parents[current]
    channels=[('root',x) for x in root['order']]
    for b in bones:channels.extend((b['name'][0],x) for x in b.get('dof',[]))
    return dict(bones=bones,bone_count=len(bones),channels=channels,
        expected_joint_dofs={'root':6,**{b['name'][0]:len(b.get('dof',[])) for b in bones if b.get('dof')}},
        root_length_to_meters=.0254/scale,angles='degrees',hierarchy_edges=edges,
        editing_only_bones=[n for n in named if 'finger' in n.lower() or 'thumb' in n.lower()])


def parse_amc(raw,skeleton):
    lines=raw.decode('ascii').splitlines();frames=[];current=None;number=None;directives=[]
    def finish():
        if current is None:return
        need(set(current)==set(skeleton['expected_joint_dofs']),'AMC frame joint set differs from calibrated ASF')
        need(number==len(frames)+1,'AMC frames not complete sequential original frames')
        frames.append([current[joint][list_channel] for joint,list_channel in indices])
    indices=[]
    cursor={}
    for joint,_ in skeleton['channels']:
        indices.append((joint,cursor.get(joint,0)));cursor[joint]=cursor.get(joint,0)+1
    for original in lines:
        line=original.partition('#')[0].strip()
        if not line:continue
        if line.startswith(':'):directives.append(line);continue
        if line.isdecimal():
            finish();number=int(line);current={};continue
        need(current is not None,'AMC channel before frame')
        parts=line.split();name=parts[0]
        need(name not in current and name in skeleton['expected_joint_dofs'],'duplicate/unknown AMC joint')
        values=numbers(parts[1:]);need(len(values)==skeleton['expected_joint_dofs'][name],'AMC channel count differs from ASF')
        current[name]=values
    finish()
    need(':FULLY-SPECIFIED' in directives and ':DEGREES' in directives and len(frames)>0,'full degree-valued AMC motion required')
    return frames
