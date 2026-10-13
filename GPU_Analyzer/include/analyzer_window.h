// GPT6-Astra: native monitor window, independent of Trainer and TUI lifetime.
#pragma once
#include "gpu_telemetry_protocol.h"
void analyzer_window_sample(const GpuTelemetrySample* sample);
void analyzer_request_stop();
