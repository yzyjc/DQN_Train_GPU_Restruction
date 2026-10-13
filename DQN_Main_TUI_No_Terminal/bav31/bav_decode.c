/* BAV 3.1-r1 — decoder core (spec §3 and §5).
 *
 * Platform-free on purpose: no files, no console, no audio.  The caller hands
 * in a memory buffer (a .bav read from disk, or one sealed into the image by
 * bav_hdr) and gets back one frame at a time plus that frame's PCM slice.
 *
 * It implements the §5.1 checklist in order, §5.2 resynchronisation (sync word
 * + CRC + cross-confirmation against the following unit), and never lets a
 * damaged unit corrupt the reference frame or shift the timeline (§5.3).
 */
#include "bav.h"

#include <stdlib.h>
#include <string.h>

static uint32_t rd16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p)
{
    uint64_t v = 0;
    int i;
    for (i = 7; i >= 0; i--) { v = (v << 8) | (uint64_t)p[i]; }
    return v;
}

const char *bav_strerror(int code)
{
    switch (code) {
    case BAV_OK:          return "ok";
    case BAV_ERR_SHORT:   return "data shorter than the 60-byte header";
    case BAV_ERR_MAGIC:   return "not a BAV file (magic is not \"BAV3\")";
    case BAV_ERR_VERSION: return "unsupported version (need 3.1)";
    case BAV_ERR_HEADER:  return "malformed header (size/flags/reserved)";
    case BAV_ERR_CRC:     return "header_crc32 mismatch";
    case BAV_ERR_FORMAT:  return "unsupported or inconsistent format fields";
    case BAV_ERR_MEM:     return "out of memory";
    default:              return "unknown error";
    }
}

/* ---------------- one unit: size + CRC, nothing decoded ---------------- */
/* Returns 1 when `at` starts a unit that passes the §5.1 size/reserved/CRC
 * checks.  Used both by the normal path and by the resync scanner. */
static int unit_probe(const bav_decoder *d, size_t at,
                      uint32_t *out_vs, uint32_t *out_as, size_t *out_total)
{
    const uint8_t *p;
    uint32_t vs, as, crc, want;

    if (at + BAV_UNIT_HEADER_SIZE > d->len) { return 0; }
    p = d->data + at;
    if (memcmp(p, BAV_UNIT_SYNC, 4) != 0) { return 0; }
    if (p[5] != 0u || rd16(p + 6) != 0u) { return 0; }
    vs = rd32(p + 8);
    as = rd32(p + 12);
    if ((uint64_t)at + BAV_UNIT_HEADER_SIZE + vs + as > (uint64_t)d->len) { return 0; }
    /* A GRAY10 payload can never need more than 4 bytes per pixel (a run record
     * per pixel) plus the flags and run_count.  This keeps a corrupt size field
     * from dragging the decoder into a huge read. */
    if ((uint64_t)vs > (uint64_t)d->width * d->height * 4u + 4096u) { return 0; }
    if (d->channels == 0u) {
        if (as != 0u) { return 0; }
    }
    crc = bav_crc32_update(bav_crc32_init(), p, 16u);
    if (vs != 0u) { crc = bav_crc32_update(crc, p + BAV_UNIT_HEADER_SIZE, vs); }
    if (as != 0u) { crc = bav_crc32_update(crc, p + BAV_UNIT_HEADER_SIZE + vs, as); }
    want = bav_crc32_final(crc);
    if (want != rd32(p + 16)) { return 0; }

    if (out_vs) { *out_vs = vs; }
    if (out_as) { *out_as = as; }
    if (out_total) { *out_total = BAV_UNIT_HEADER_SIZE + (size_t)vs + (size_t)as; }
    return 1;
}

/* §5.2: byte-scan forward for the next position where a unit validates *and*
 * the unit after it validates too.  The sentinel alone is not enough - PCM
 * contains "BAVF" by chance roughly once every 6.8 hours of stereo audio. */
static int resync(bav_decoder *d)
{
    size_t at;

    for (at = d->off + 1u; at + BAV_UNIT_HEADER_SIZE <= d->len; at++) {
        uint32_t vs = 0, as = 0;
        size_t total = 0, next_total = 0;
        if (!unit_probe(d, at, &vs, &as, &total)) { continue; }
        if (at + total == d->len) { d->off = at; d->resyncs++; return 1; }
        if (unit_probe(d, at + total, NULL, NULL, &next_total)) {
            d->off = at;
            d->resyncs++;
            return 1;
        }
    }
    return 0;
}

/* ---------------- open / validate the header (spec §5.4) ---------------- */

int bav_dec_open(bav_decoder *d, const void *data, size_t len)
{
    const uint8_t *h;

    if (d == NULL || data == NULL) { return BAV_ERR_SHORT; }
    memset(d, 0, sizeof *d);
    d->data = (const uint8_t *)data;
    d->len = len;

    if (len < BAV_HEADER_SIZE) { return BAV_ERR_SHORT; }
    h = (const uint8_t *)data;
    if (memcmp(h, BAV_MAGIC, 4) != 0) { return BAV_ERR_MAGIC; }
    if (h[4] != BAV_VERSION_MAJOR || h[5] != BAV_VERSION_MINOR) { return BAV_ERR_VERSION; }
    if (rd16(h + 6) != BAV_HEADER_SIZE) { return BAV_ERR_HEADER; }
    if (rd32(h + 8) != 0u) { return BAV_ERR_HEADER; }
    if (bav_crc32(h, 56) != rd32(h + 56)) { return BAV_ERR_CRC; }

    d->width  = rd16(h + 18);
    d->height = rd16(h + 20);
    d->fps_num = rd32(h + 22);
    d->fps_den = rd32(h + 26);
    d->frame_count = rd32(h + 30);
    d->channels = h[14];
    d->sample_rate = rd32(h + 34);
    d->bytes_per_sample = h[16];
    d->total_sample_frames = rd64(h + 38);

    if (rd16(h + 12) != BAV_VF_GRAY10) { return BAV_ERR_FORMAT; }
    if (d->width == 0u || d->height == 0u ||
        d->width > BAV_GRAY10_MAX_DIM || d->height > BAV_GRAY10_MAX_DIM) { return BAV_ERR_FORMAT; }
    if (d->fps_num == 0u || d->fps_den == 0u) { return BAV_ERR_FORMAT; }
    if (d->frame_count == 0u) { return BAV_ERR_FORMAT; }
    if (d->channels > 2u) { return BAV_ERR_FORMAT; }
    if ((d->channels == 0u) != (d->sample_rate == 0u)) { return BAV_ERR_FORMAT; }
    if (d->channels != 0u) {
        if (h[15] != BAV_ASF_S16LE || d->bytes_per_sample != 2u) { return BAV_ERR_FORMAT; }
        if (d->total_sample_frames != bav_a_n(d->frame_count, d->sample_rate,
                                              d->fps_num, d->fps_den)) {
            return BAV_ERR_FORMAT;
        }
    }

    d->fb = (uint8_t *)calloc((size_t)d->width * d->height, 1);
    if (d->fb == NULL) { return BAV_ERR_MEM; }
    d->off = BAV_HEADER_SIZE;
    d->index = 0u;
    return BAV_OK;
}

/* ---------------- decode one GRAY10 payload into the framebuffer ------- */

static int decode_video(bav_decoder *d, const uint8_t *p, uint32_t vs)
{
    const uint8_t *end = p + vs;
    uint8_t ff;
    uint32_t w = d->width, h = d->height, r;

    if (vs == 0u) { return 0; }
    ff = *p++;
    if ((ff & ~(BAV_FF_IS_DIFF | BAV_FF_HAS_RECT)) != 0u) { return 0; }
    if (!(ff & BAV_FF_IS_DIFF)) { memset(d->fb, 0, (size_t)w * h); }

    if (ff & BAV_FF_HAS_RECT) {
        uint32_t rc;
        if (p >= end) { return 0; }
        rc = *p++;
        for (r = 0; r < rc; r++) {
            uint32_t x, y, rw, rh, val, yy;
            if (p + 5 > end) { return 0; }
            x = p[0]; y = p[1]; rw = p[2]; rh = p[3]; val = p[4];
            p += 5;
            if (x >= w || y >= h || rw == 0u || rh == 0u ||
                x + rw > w || y + rh > h || val > 9u) { return 0; }
            for (yy = y; yy < y + rh; yy++) {
                memset(d->fb + (size_t)yy * w + x, (int)val, rw);
            }
        }
    }
    {
        uint32_t segs;
        if (p + 2 > end) { return 0; }
        segs = rd16(p); p += 2;
        for (r = 0; r < segs; r++) {
            uint32_t row, col, len, k;
            int delta;
            if (p + 4 > end) { return 0; }
            row = p[0]; col = p[1]; len = p[2];
            delta = (int)(int8_t)(uint8_t)(p[3] - 9u);
            p += 4;
            if (row >= h || len == 0u || col >= w || col + len > w) { return 0; }
            for (k = 0; k < len; k++) {
                uint8_t *q = &d->fb[(size_t)row * w + col + k];
                int v = (ff & BAV_FF_IS_DIFF) ? ((int)*q + delta) : delta;
                if (v < 0) { v = 0; }
                if (v > 9) { v = 9; }
                *q = (uint8_t)v;
            }
        }
    }
    if (p != end) { return 0; }   /* §3.3: payload must be consumed exactly */
    return 1;
}

int bav_dec_next(bav_decoder *d, const uint8_t **audio, size_t *audio_len)
{
    int guard = 0;

    if (audio) { *audio = NULL; }
    if (audio_len) { *audio_len = 0; }
    if (d == NULL || d->fb == NULL) { return 0; }
    if (d->index >= d->frame_count) { return 0; }

    for (;;) {
        const uint8_t *uh, *vp, *ap;
        uint32_t vs = 0, as = 0;
        size_t total = 0;

        if (++guard > 64) { d->truncated = 1; return 0; }

        if (!unit_probe(d, d->off, &vs, &as, &total)) {
            goto lose_one;
        }
        uh = d->data + d->off;
        vp = uh + BAV_UNIT_HEADER_SIZE;
        ap = vp + vs;

        /* §5.1 items unit_probe does not cover: flag sanity and the
         * "empty payload <=> NO_VIDEO_UPDATE" equivalence. */
        if ((uh[4] & ~(BAV_UF_KEYFRAME | BAV_UF_NO_VIDEO_UPDATE)) != 0u ||
            ((uh[4] & BAV_UF_NO_VIDEO_UPDATE) != 0u) != (vs == 0u)) {
            goto lose_one;
        }
        if (d->channels != 0u) {
            uint64_t n = bav_dec_frame_samples(d, d->index);
            if ((uint64_t)as != n * d->channels * d->bytes_per_sample) {
                goto lose_one;              /* §4.5: the size is normative */
            }
        }

        /* Audio first: a damaged *video* payload must not cost us the audio
         * (spec §5.3, audio continuity has priority). */
        if (audio) { *audio = (as != 0u) ? ap : NULL; }
        if (audio_len) { *audio_len = as; }

        if (vs != 0u && !decode_video(d, vp, vs)) {
            d->errors++;                    /* hold the previous frame */
        }
        d->off += total;
        d->index++;
        return 1;

lose_one:
        /* One unit is gone.  The unit header carries no frame index, so the
         * decoder cannot know how many frames a jump skipped: it accounts for
         * exactly one, advances the timeline normally, and hands the caller the
         * silence needed to conceal it.  The audio clock (which comes from the
         * device, spec §4.4) is never touched. */
        d->errors++;
        d->lost_frames++;
        if (d->channels != 0u) {
            d->pending_silence += bav_dec_frame_samples(d, d->index);
        }
        d->index++;
        if (d->index >= d->frame_count) { d->truncated = 1; return 0; }
        d->off += 1u;
        if (!resync(d)) { d->truncated = 1; return 0; }
        /* keep looking for a valid unit for the new index */
    }
}

const uint8_t *bav_dec_pixels(const bav_decoder *d)
{
    return (d != NULL) ? d->fb : NULL;
}

uint64_t bav_dec_frame_samples(const bav_decoder *d, uint32_t n)
{
    if (d->channels == 0u || d->sample_rate == 0u) { return 0u; }
    return bav_a_n((uint64_t)n + 1u, d->sample_rate, d->fps_num, d->fps_den)
         - bav_a_n((uint64_t)n,      d->sample_rate, d->fps_num, d->fps_den);
}

double bav_dec_frame_start(const bav_decoder *d, uint32_t n)
{
    if (d->channels == 0u || d->sample_rate == 0u) {
        return (double)n * d->fps_den / d->fps_num;
    }
    return (double)bav_a_n(n, d->sample_rate, d->fps_num, d->fps_den)
         / (double)d->sample_rate;
}

double bav_dec_display_time(const bav_decoder *d, uint32_t n)
{
    if (d->channels == 0u || d->sample_rate == 0u) {
        return ((double)n + 0.5) * d->fps_den / d->fps_num;
    }
    /* (A_n + N_n/2) / Fs  ==  (A_n + A_(n+1)) / (2 Fs) */
    return ((double)bav_a_n(n, d->sample_rate, d->fps_num, d->fps_den)
          + (double)bav_a_n((uint64_t)n + 1u, d->sample_rate, d->fps_num, d->fps_den))
         / (2.0 * (double)d->sample_rate);
}

void bav_dec_close(bav_decoder *d)
{
    if (d == NULL) { return; }
    free(d->fb);
    d->fb = NULL;
}
