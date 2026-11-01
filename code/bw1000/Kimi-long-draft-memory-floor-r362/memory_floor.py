"""Release inactive allocator cache once at pressure; retain the original floor."""
import time

FLOOR = 4 << 30


def guarded_codec(base, torch, report):
    class GuardedCodec(base):
        def fence(self):
            super().fence()
            cuda = torch.cuda
            free, total = cuda.mem_get_info()
            row = dict(index=len(report.setdefault('memory_fences', [])),
                       free_before_bytes=free, total_bytes=total,
                       allocated_before_bytes=cuda.memory_allocated(),
                       reserved_before_bytes=cuda.memory_reserved(), trimmed=False)
            # Keep pressure diagnostics even when trimming or the unchanged floor fails.
            report['memory_fences'].append(row)
            report['minimum_observed_free_draft_bytes'] = min(
                report.get('minimum_observed_free_draft_bytes', free), free)
            if free < FLOOR:
                row['trimmed'] = True
                report['allocator_trim_count'] = report.get('allocator_trim_count', 0) + 1
                start = time.perf_counter()
                try:
                    cuda.empty_cache()
                    cuda.synchronize()
                finally:
                    row['trim_wall_s'] = time.perf_counter() - start
                free, total = cuda.mem_get_info()
            row.update(free_after_bytes=free, allocated_after_bytes=cuda.memory_allocated(),
                       reserved_after_bytes=cuda.memory_reserved(), floor_bytes=FLOOR)
            report['peak_allocated_draft_bytes'] = cuda.max_memory_allocated()
            report['peak_reserved_draft_bytes'] = cuda.max_memory_reserved()
            report['minimum_free_draft_bytes'] = min(report.get('minimum_free_draft_bytes', free), free)
            if free < FLOOR:
                row['floor_passed'] = False
                raise RuntimeError('Original4GiB draft memory floor')
            row['floor_passed'] = True
    return GuardedCodec
