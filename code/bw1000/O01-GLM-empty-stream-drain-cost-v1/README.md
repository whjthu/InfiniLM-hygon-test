# Original GLM SDK empty stream drain diagnostic

One finite GPU step on O01 f12r3n17 under parent1087656. Use the exact original906 CUDA-compat runtime SHA ac547071... and a dedicated empty nonblocking stream. Compare one and two synchronous calls for OFFON/ONOFF/OFFON three pairs, each five warm windows and20 measured windows of512 units. All150 windows and115200 timed SDK calls are retained. No kernels, tensors, weights, Native model or Torch import occurs.

This measures the already-drained SDK call path including Python/ctypes overhead. It does not measure Runtime::syncStream wrapper overhead, Tensor destruction, pending work, original1.07ms adjacent event window, Model throughput, GPUbusy or numerical math. No Model or optimization promotion follows from this cost alone.

Reuses the previous qualified finite receiver and independent pre/post PCI/KFD snapshot pattern. Source and label are immutable, pre-existing label or namespace aborts without replay. One original360s epoch covers CPUpre55/GPU145/CPUpost55 and accounting/cleanup. Worker deadline is originalepoch+140, never reset. Unknown transport is reconciled through original intent/accounting, not replayed. No new service or poller.
