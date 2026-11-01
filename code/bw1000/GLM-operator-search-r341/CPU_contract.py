import ast,pathlib,json,struct,math
p=pathlib.Path(__file__).parent
for f in p.glob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
checks=0
for lanes in [16,32,64,128]:
 for groups in [1,2,4,8]:
  assert lanes*groups<=1024
  for rows in [1,8,64,128,4096]:
   blocks=rows*2;covered=set()
   for block in range((blocks+groups-1)//groups):
    for group in range(groups):
     b=block*groups+group
     if b<blocks:
      assert b not in covered;covered.add(b)
   assert covered==set(range(blocks))
  assert {lane+j*lanes for lane in range(lanes) for j in range(128//lanes)}==set(range(128));checks+=1
def decode(v):
 e,m=(v>>3)&15,v&7
 return math.ldexp(m,-9) if not e else math.ldexp(1+m/8,e-7)
def encode(a):
 a=min(abs(a),448.)
 if a<.015625:return round(a*512)
 e=math.floor(math.log2(a));m=round(math.ldexp(a,3-e))
 if m==16:e+=1;m=8
 return (e+7)*8+m-8
values=[decode(v) for v in range(127)]
for i,x in enumerate(values):
 assert encode(x)==i
 if i<126:
  y=(x+values[i+1])/2;assert encode(y)==(i if i%2==0 else i+1)
  for z in [math.nextafter(y,0),math.nextafter(y,math.inf)]:
   expect=min(range(127),key=lambda k:(abs(values[k]-z),k%2));assert encode(z)==expect
print(json.dumps({'CPU_only':True,'AST':True,'launch_variants':checks,'FP8_all_finite_positive_codes_and_ties':True,'tail_coverage':True,'GPU_not_initialized':True}))
