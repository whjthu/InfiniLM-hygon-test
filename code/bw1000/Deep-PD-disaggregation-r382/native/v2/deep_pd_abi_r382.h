#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
const char *deep_pd_state_abi_r382(void);
uint64_t deep_pd_state_generation_r382(void *ctx);
int deep_pd_state_begin_export_r382(void *ctx, uint64_t past, uint64_t epoch, uint64_t *sid);
const char *deep_pd_state_descriptor_r382(void *ctx, uint64_t sid);
int deep_pd_state_read_r382(void *ctx, uint64_t sid, uint32_t shard, uint32_t block,
                          uint64_t offset, void *hostdst, uint64_t n);
int deep_pd_state_begin_import_r382(void *ctx, const char *manifest_json, uint64_t epoch,
                                  uint64_t expected_target_generation, uint64_t *sid);
int deep_pd_state_write_r382(void *ctx, uint64_t sid, uint32_t shard, uint32_t block,
                           uint64_t offset, const void *hostsrc, uint64_t n);
int deep_pd_state_validate_r382(void *ctx, uint64_t sid);
int deep_pd_state_commit_r382(void *ctx, uint64_t sid);
int deep_pd_state_abort_r382(void *ctx, uint64_t sid);
int deep_pd_state_poison_r382(void *ctx, uint64_t expected_generation);
#ifdef __cplusplus
}
#endif
