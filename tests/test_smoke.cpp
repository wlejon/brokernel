// The library links, and brass's SPIR-V target is reachable through it.
#include "brokernel/brokernel.h"

#include <brass/target/spirv_target.hpp>

#include <cstdio>
#include <cstring>

int main() {
    const char* v = brokernel::version();
    if (!v || std::strlen(v) == 0) {
        std::fprintf(stderr, "test_smoke: empty version\n");
        return 1;
    }
    [[maybe_unused]] const brass::target::SpirvStage stage = brass::target::SpirvStage::Compute;
    std::printf("test_smoke: brokernel %s\n", v);
    return 0;
}
