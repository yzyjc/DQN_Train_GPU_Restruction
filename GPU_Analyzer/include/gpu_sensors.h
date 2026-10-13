// GPT6-Astra: optional read-only ADLX backend; absent/invalid readings stay -1.
#pragma once
#include "gpu_telemetry_protocol.h"
void* gpu_sensors_create(const char* expected_name);
void gpu_sensors_sample(void* sensors, GpuTelemetrySample* sample);
void gpu_sensors_destroy(void* sensors);
