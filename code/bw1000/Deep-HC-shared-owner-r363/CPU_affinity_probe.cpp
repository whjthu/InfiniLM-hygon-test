// CPU only: compile the exact production helper, including its thread-local state.
#include "execution/owner_affinity_r2392.hpp"
#include <cstring>
extern "C" int probe(int device, int *selected, unsigned *changes, char *error) {
    try {
        infinilm::execution::affinity_r2392::apply_once(device);
        cpu_set_t mask{};
        if (pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask))
            throw std::runtime_error("probe readback failed");
        int count = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask)) selected[count++] = cpu;
        *changes = infinilm::execution::affinity_r2392::changes.at(device).load();
        return count;
    } catch (const std::exception &e) {
        std::strncpy(error, e.what(), 1023);
        error[1023] = 0;
        return -1;
    }
}
