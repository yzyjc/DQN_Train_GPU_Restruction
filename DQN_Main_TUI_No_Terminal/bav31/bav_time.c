/* BAV 3.1-r1 — the integer timing model (spec §4.2 and §2.7).
 *
 * Shared by BOTH sides: the encoder needs the sample boundaries to slice PCM
 * and the keyframe distance to stay conformant, the decoder needs them to
 * compute the display time.  That is why this lives in its own module instead
 * of hiding inside the encoder.
 *
 * Everything here is pure integer arithmetic - no floating point anywhere,
 * because A_n must be reproducible bit-for-bit by every implementation.
 */
#include "bav.h"

uint64_t bav_a_n(uint64_t n, uint32_t fs, uint32_t num, uint32_t den)
{
    uint64_t q, r;

    if (num == 0u) {
        return 0u;
    }
    /* floor(n*fs*den/num) without ever forming n*fs*den:
     *   n = q*num + r  =>  A_n = q*fs*den + floor(r*fs*den/num)
     * q*fs*den stays far inside 64 bits for every field range the spec allows. */
    q = n / num;
    r = n % num;
    return q * (uint64_t)fs * (uint64_t)den
         + ((r * (uint64_t)fs * (uint64_t)den) / num);
}

uint64_t bav_frame_gap_max(uint32_t num, uint32_t den)
{
    uint64_t g;

    if (den == 0u) {
        return 1u;
    }
    g = (2u * (uint64_t)num) / (uint64_t)den;
    return (g < 1u) ? 1u : g;   /* FPS < 0.5: rule degenerates to "every frame" */
}

uint64_t bav_total_sample_frames(uint32_t frame_count, uint32_t fs,
                                 uint32_t num, uint32_t den)
{
    return bav_a_n((uint64_t)frame_count, fs, num, den);
}
