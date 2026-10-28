#pragma once
// Common-scale path constructs exactly representable normal FP32 E2M1 values.
// Edge scales use original ldexpf, preserving subnormal/overflow/signedzero semantics.
template<class FromBits,class Original>
inline __attribute__((always_inline)) void fp4DirectPairR161(unsigned char packed,unsigned char scale,float&lo,float&hi,FromBits from_bits,Original original){
 if(scale<2||scale>252){original(packed,scale,lo,hi);return;}
 auto decode=[&](unsigned code){const unsigned m=code&7,sign=(code&8)<<28;const unsigned exponent=unsigned(scale)+(m>>1)-1;const unsigned bits=m==0?sign:sign|(exponent<<23)|(m==1?0u:(m&1)<<22);return from_bits(bits);};
 lo=decode(packed&15);hi=decode(packed>>4);
}
