// 一次性诊断：确认 ADL 在 8060S 上哪一步失败
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <string.h>

typedef struct ADLThermalControllerInfo {
    int iSize;
    int iThermalDomain;
    int iTemperature;
    int iMinTemperature;
    int iMaxTemperature;
} ADLThermalControllerInfo;
typedef struct ADLTemperature { int iSize; int iTemperature; } ADLTemperature;

int main(void) {
    printf("=== ADL diagnostic on AMD 8060S ===\n");

    HMODULE adl = LoadLibraryA("atiadlxx.dll");
    printf("[1] LoadLibraryA(atiadlxx.dll) -> %p\n", (void*)adl);
    if (!adl) {
        printf("    GetLastError: %lu\n", GetLastError());
        printf("RESULT: ADL_DLL_MISSING\n");
        return 1;
    }

    auto create = (int(*)(void))GetProcAddress(adl, "ADL_Main_Control_Create");
    printf("[2] ADL_Main_Control_Create addr -> %p\n", (void*)create);
    if (!create) {
        printf("    GetLastError: %lu\n", GetLastError());
        printf("RESULT: CREATE_ENTRY_MISSING\n");
        return 2;
    }
    int rc = create();
    printf("[3] ADL_Main_Control_Create() -> %d  (0 = success)\n", rc);
    if (rc != 0) {
        printf("RESULT: ADL_CREATE_FAILED rc=%d\n", rc);
        return 3;
    }

    auto nAdapters = (int(*)(int*))GetProcAddress(adl, "ADL_Adapter_NumberOfAdapters_Get");
    auto getThermal = (int(*)(int, int, ADLThermalControllerInfo*))GetProcAddress(adl, "ADL_Adapter_GetThermalStatus");
    auto getTempOv5 = (int(*)(int, int, ADLTemperature*))GetProcAddress(adl, "ADL_Overdrive5_Temperature_Get");
    printf("[4] NumberOfAdapters=%p GetThermalStatus=%p Ov5TempGet=%p\n",
        (void*)nAdapters, (void*)getThermal, (void*)getTempOv5);

    int count = -1;
    if (nAdapters) {
        int rc2 = nAdapters(&count);
        printf("[5] ADL_Adapter_NumberOfAdapters_Get -> rc=%d count=%d\n", rc2, count);
    }

    int found = 0;
    if (getThermal) {
        for (int tc = 0; tc < 8; tc++) {
            ADLThermalControllerInfo info; memset(&info, 0, sizeof(info));
            info.iSize = sizeof(info);
            int r = getThermal(0, tc, &info);
            if (r == 0 && info.iTemperature > 0) {
                printf("[6] GetThermalStatus(adapter=0, tc=%d) -> rc=%d temp=%d (%d.%03d C)\n",
                    tc, r, info.iTemperature, info.iTemperature / 1000, info.iTemperature % 1000);
                found = 1;
            } else if (r != 0) {
                printf("[6] GetThermalStatus(adapter=0, tc=%d) -> rc=%d\n", tc, r);
            }
        }
    }
    if (!found && getTempOv5) {
        for (int tc = 0; tc < 8; tc++) {
            ADLTemperature temp; memset(&temp, 0, sizeof(temp));
            temp.iSize = sizeof(temp);
            int r = getTempOv5(0, tc, &temp);
            if (r == 0 && temp.iTemperature > 0) {
                printf("[7] Overdrive5_Temperature_Get(0, %d) -> rc=%d temp=%d (%d.%03d C)\n",
                    tc, r, temp.iTemperature, temp.iTemperature / 1000, temp.iTemperature % 1000);
                found = 1;
            } else if (r != 0) {
                printf("[7] Overdrive5_Temperature_Get(0, %d) -> rc=%d\n", tc, r);
            }
        }
    }
    printf("RESULT: %s\n", found ? "TEMP_FOUND" : "TEMP_NOT_FOUND");

    auto destroy = (int(*)(void))GetProcAddress(adl, "ADL_Main_Control_Destroy");
    if (destroy) destroy();
    FreeLibrary(adl);
    return found ? 0 : 4;
}
