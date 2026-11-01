#include "original_math.cuh"
using B=__nv_bfloat16;
__device__ float sum64(float x){for(int d=32;d;d/=2)x+=__shfl_xor_sync(~0ull,x,d,64);return x;}
__device__ float sig(float x){return x>=0?1.f/(1.f+expf(-x)):expf(x)/(1.f+expf(x));}
__global__ void prepare(const B*q,const B*k,const B*g,const float*a,const float*bias,float*p){
 int h=blockIdx.x,j=threadIdx.x;float q0=float(q[h*128+j]),q1=float(q[h*128+j+64]),k0=float(k[h*128+j]),k1=float(k[h*128+j+64]);
 float qs=rsqrtf(sum64(q0*q0+q1*q1)+1e-6f)*0.08838834764831844f,ks=rsqrtf(sum64(k0*k0+k1*k1)+1e-6f);
 for(int z=0;z<2;z++){int i=h*128+j+64*z;p[i]=float(q[i])*qs;p[768+i]=float(k[i])*ks;p[1536+i]=expf(-5.f*sig(expf(a[h])*(float(g[i])+bias[i])));}
}
template<int W,bool P> __global__ void update(B*s,const B*q,const B*k,const B*v,const B*g,const B*b,const float*a,const float*bias,B*out,const float*p){
 int lane=threadIdx.x%64,row=blockIdx.x*W+threadIdx.x/64;if(row>=768)return;int h=row/128;
 float qv[2],kv[2],dec[2],st[2];float qs=0,ks=0;
 #pragma unroll
 for(int z=0;z<2;z++){int i=h*128+lane+64*z;qv[z]=P?p[i]:float(q[i]);kv[z]=P?p[768+i]:float(k[i]);qs+=qv[z]*qv[z];ks+=kv[z]*kv[z];}
 if(!P){qs=rsqrtf(sum64(qs)+1e-6f)*0.08838834764831844f;ks=rsqrtf(sum64(ks)+1e-6f);}
 float mem=0,hq=0,kq=0;
 #pragma unroll
 for(int z=0;z<2;z++){int i=h*128+lane+64*z;if(!P){qv[z]*=qs;kv[z]*=ks;}dec[z]=P?p[1536+i]:expf(-5.f*sig(expf(a[h])*(float(g[i])+bias[i])));st[z]=float(s[98304+row*128+lane+64*z])*dec[z];mem+=st[z]*kv[z];hq+=st[z]*qv[z];kq+=kv[z]*qv[z];}
 float delta=(float(v[row])-sum64(mem))*sig(float(b[h]));hq=sum64(hq);kq=sum64(kq);if(lane==0)out[row]=B(hq+delta*kq);
 #pragma unroll
 for(int z=0;z<2;z++)s[196608+row*128+lane+64*z]=B(st[z]+kv[z]*delta);
}
extern "C" int search_run(int arm,void**p,void*stream){
 auto st=(B*)p[0];auto q=(B*)p[1];auto k=(B*)p[2];auto v=(B*)p[3];auto g=(B*)p[4];auto b=(B*)p[5];auto a=(float*)p[6];auto bias=(float*)p[7];auto out=(B*)p[8];auto snap=(B*)p[9];auto tmp=(float*)p[13];auto s=(cudaStream_t)stream;
 if(arm==0){auto e=cudaMemcpyAsync(snap,st,3*98304*sizeof(B),cudaMemcpyDeviceToDevice,s);if(e!=cudaSuccess)return int(e);
 kimiDeltaAttentionRecurrentCudaKernel<B,B><<<dim3(1,6,128),256,(384+256)*4,s>>>(out,snap,nullptr,st,q,k,v,g,b,a,bias,p[10],p[11],p[12],false,false,false,true,true,true,1,128,3,0.08838834764831844f,-5.f,768,768,128,98304,16384,128,1,0,0,0,0,768,768,128,768,768,128,768,768,128,768,768,128,6,6,1,1,128);
 }else{bool prep=arm>4;int w=(arm-1)%4;if(prep)prepare<<<6,64,0,s>>>(q,k,g,a,bias,tmp);
 #define GO(N) if(prep)update<N,true><<<(768+N-1)/N,N*64,0,s>>>(st,q,k,v,g,b,a,bias,out,tmp);else update<N,false><<<(768+N-1)/N,N*64,0,s>>>(st,q,k,v,g,b,a,bias,out,tmp)
 switch(w){case 0:GO(1);break;case 1:GO(2);break;case 2:GO(4);break;case 3:GO(8);break;}
 }return int(cudaGetLastError());
}
