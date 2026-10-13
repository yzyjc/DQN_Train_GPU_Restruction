#pragma once

// Accelerator Manager
// Auto-detect hardware and load optimal accelerator DLL

#include "accelerator.h"

#ifdef __cplusplus
extern "C" {
#endif

// Manager handle
typedef void* AccelManagerHandle;

// Initialize manager
AccelStatus accel_manager_init(AccelManagerHandle* out_manager);

// Cleanup manager
AccelStatus accel_manager_cleanup(AccelManagerHandle manager);

// Auto-detect and create best accelerator
AccelStatus accel_manager_create_auto(AccelManagerHandle manager, AcceleratorHandle* out_accel);

// Create specific accelerator type
AccelStatus accel_manager_create(AccelManagerHandle manager, AccelType type, AcceleratorHandle* out_accel);

// Get available accelerator types
AccelStatus accel_manager_get_available(AccelManagerHandle manager, AccelType* types, int max_count, int* out_count);

// Get accelerator name
const char* accel_manager_get_name(AccelType type);

#ifdef __cplusplus
}
#endif
