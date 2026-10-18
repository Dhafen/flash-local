// Bounded allocator ownership control, with no model or Metal device needed.
#include "ggml-impl.h"
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static bool unmapped(uintptr_t pointer, size_t size) {
    mach_vm_address_t address = pointer;
    mach_vm_size_t extent = 0;
    vm_region_basic_info_data_64_t information{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto status = mach_vm_region(mach_task_self(), &address, &extent, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&information), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    return status == KERN_INVALID_ADDRESS || (status == KERN_SUCCESS && address >= pointer + size);
}
int main() {
    try {
        const size_t page = sysconf(_SC_PAGESIZE);
        require(page > 0, "System page size unavailable");
        for (size_t size : {page + 1, (size_t(1) << 20) + 1, size_t(64) << 20}) {
            // Repeat to catch a mismatched allocation/free family. Touch every
            // page so this tests committed memory, not just reserved addresses.
            for (int iteration = 0; iteration < 3; ++iteration) {
                void * buffer = ggml_aligned_malloc(size);
                require(buffer && uintptr_t(buffer) % page == 0, "VM buffer not page aligned");
                volatile uint8_t * touched = static_cast<volatile uint8_t *>(buffer);
                for (size_t offset = 0; offset < size; offset += page) touched[offset] = 0xa5;
                touched[size - 1] = 0x5a;
                const uintptr_t address = uintptr_t(buffer);
                const bool allocated = !unmapped(address, size);
                ggml_aligned_free(buffer, size);
                require(allocated && unmapped(address, size), "Freed VM buffer mapping remains live");
            }
        }
        ggml_aligned_free(nullptr, 0);
        std::cout << "ORNITH_VM_ALLOCATION pageAligned=true touchedMappingsReleased=true cycles=9 modelLoaded=false physicalPhone=false\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
