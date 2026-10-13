/* CRC-32/ISO-HDLC (zlib / PKZIP / PNG) — spec §0.7.
 * poly 0x04C11DB7 reflected = 0xEDB88320, init/xorout = 0xFFFFFFFF.
 */
#include "bav.h"

static uint32_t s_table[256];
static int      s_table_ready;

static void build_table(void)
{
    uint32_t i, k;
    for (i = 0; i < 256u; i++) {
        uint32_t c = i;
        for (k = 0; k < 8u; k++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        s_table[i] = c;
    }
    s_table_ready = 1;
}

uint32_t bav_crc32_init(void)
{
    if (!s_table_ready) {
        build_table();
    }
    return 0xFFFFFFFFu;
}

uint32_t bav_crc32_update(uint32_t state, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t i;

    if (!s_table_ready) {
        build_table();
    }
    for (i = 0; i < len; i++) {
        state = s_table[(state ^ p[i]) & 0xFFu] ^ (state >> 8);
    }
    return state;
}

uint32_t bav_crc32_final(uint32_t state)
{
    return state ^ 0xFFFFFFFFu;
}

uint32_t bav_crc32(const void *data, size_t len)
{
    return bav_crc32_final(bav_crc32_update(bav_crc32_init(), data, len));
}
