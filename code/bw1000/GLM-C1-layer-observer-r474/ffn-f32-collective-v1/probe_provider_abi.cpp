#include <infiniccl/infiniccl.h>

template <int, int, unsigned long> struct Probe {};
using Observed = Probe<infinicclFloat32, infinicclSum, sizeof(infinicclUniqueId)>;
void capture(Observed) {}
