#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All pointers belong to the caller's same Hygon device and stream lease.
// status is caller-owned aligned4B UInt8[4], zeroed once for a complete query.
// No allocation, device switch, synchronization, ownership transfer or cache.
// cast direction0 BF16->F32 /1 F32->BF16, n only16384 or2048.
int glm_value_cast_r446(int device,int direction,size_t n,const void*input,void*output,uint32_t*status,void*stream);
// kind0 [1,2048]@[2048,512], kind1 [1,512]@[512,256], F32 only.
int glm_value_gemv_r446(int device,int kind,const float*x,const float*w,float*y,uint32_t*status,void*stream);
#ifdef __cplusplus
}
#endif
