/* BAV 3.1-r1 — audio output and the four clocks (spec §4.4, Windows/waveOut).
 *
 * waveOut is used deliberately:
 *   - it goes through the shared audio engine, so it mixes with whatever else
 *     the host program is playing (no exclusive takeover);
 *   - waveOutGetPosition reports the *driver's playback position*, i.e. the
 *     closest thing this API family offers to "presented to the output" -
 *     exactly the quantity spec §4.4 defines T_a from.
 *
 * The queue is a small ring of fixed chunks; the main loop polls WHDR_DONE
 * instead of using a callback, because waveOut forbids calling waveOut* from
 * the callback thread.
 */
#include "bav.h"

#include <windows.h>
#include <mmsystem.h>

#include <stdlib.h>
#include <string.h>

#define BAV_AUDIO_CHUNKS 8
#define BAV_AUDIO_CHUNK  4096

struct bav_audio {
    HWAVEOUT  h;
    WAVEHDR   hdr[BAV_AUDIO_CHUNKS];
    uint8_t  *buf[BAV_AUDIO_CHUNKS];
    int       used[BAV_AUDIO_CHUNKS];
    uint32_t  rate;
    uint32_t  channels;
    uint32_t  block;            /* bytes per sample frame */
    uint64_t  submitted;        /* sample frames handed to the API */
    uint64_t  presented;        /* sample frames the driver reports played */
    uint64_t  start_pos;        /* position read at open time */
    uint64_t  queued_bytes;
    uint64_t  presented_bytes;  /* for queue accounting */
    uint64_t  written_bytes;
    double    epoch;
    int       opened;
    uint32_t  underruns;
};

double bav_monotonic(void)
{
    LARGE_INTEGER f, c;
    static LARGE_INTEGER freq;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&f);
        freq = f;
    }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}

static void reap(bav_audio *a)
{
    int i;
    for (i = 0; i < BAV_AUDIO_CHUNKS; i++) {
        if (a->used[i] && (a->hdr[i].dwFlags & WHDR_DONE)) {
            a->used[i] = 0;
            a->queued_bytes -= a->hdr[i].dwBufferLength;
            a->hdr[i].dwBufferLength = 0;
        }
    }
}

bav_audio *bav_audio_open(uint32_t sample_rate, uint32_t channels)
{
    bav_audio *a;
    WAVEFORMATEX wf;
    MMRESULT mr;
    int i;

    if (sample_rate == 0u || channels == 0u || channels > 2u) { return NULL; }
    a = (bav_audio *)calloc(1, sizeof *a);
    if (a == NULL) { return NULL; }

    memset(&wf, 0, sizeof wf);
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = (WORD)channels;
    wf.nSamplesPerSec = sample_rate;
    wf.wBitsPerSample = 16;
    wf.nBlockAlign = (WORD)(channels * 2u);
    wf.nAvgBytesPerSec = sample_rate * wf.nBlockAlign;
    wf.cbSize = 0;

    mr = waveOutOpen(&a->h, WAVE_MAPPER, &wf, 0, 0, CALLBACK_NULL);
    if (mr != MMSYSERR_NOERROR) {
        free(a);
        return NULL;
    }
    a->opened = 1;
    for (i = 0; i < BAV_AUDIO_CHUNKS; i++) {
        a->buf[i] = (uint8_t *)malloc(BAV_AUDIO_CHUNK);
        if (a->buf[i] == NULL) { bav_audio_close(a); return NULL; }
        memset(&a->hdr[i], 0, sizeof(WAVEHDR));
        a->hdr[i].lpData = (LPSTR)a->buf[i];
        a->hdr[i].dwBufferLength = 0;
    }
    a->rate = sample_rate;
    a->channels = channels;
    a->block = channels * 2u;
    a->epoch = bav_monotonic();
    {
        MMTIME mt;
        mt.wType = TIME_SAMPLES;
        if (waveOutGetPosition(a->h, &mt, sizeof mt) == MMSYSERR_NOERROR &&
            mt.wType == TIME_SAMPLES) {
            a->start_pos = mt.u.sample;
        }
    }
    return a;
}

size_t bav_audio_submit(bav_audio *a, const void *pcm, size_t bytes)
{
    const uint8_t *p = (const uint8_t *)pcm;
    size_t accepted = 0;

    if (a == NULL || !a->opened) { return bytes; }   /* no device: pretend ok */
    reap(a);
    while (accepted < bytes) {
        int i, free_idx = -1;
        size_t room, take;
        for (i = 0; i < BAV_AUDIO_CHUNKS; i++) {
            if (!a->used[i]) { free_idx = i; break; }
        }
        if (free_idx < 0) { break; }                 /* queue full */
        room = BAV_AUDIO_CHUNK;
        take = bytes - accepted;
        if (take > room) { take = room; }
        /* keep whole sample frames together */
        take -= take % a->block;
        if (take == 0u) { break; }
        memcpy(a->buf[free_idx], p + accepted, take);
        a->hdr[free_idx].dwBufferLength = (DWORD)take;
        a->hdr[free_idx].dwFlags = 0;
        if (waveOutPrepareHeader(a->h, &a->hdr[free_idx], sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
            break;
        }
        if (waveOutWrite(a->h, &a->hdr[free_idx], sizeof(WAVEHDR)) != MMSYSERR_NOERROR) {
            waveOutUnprepareHeader(a->h, &a->hdr[free_idx], sizeof(WAVEHDR));
            break;
        }
        a->used[free_idx] = 1;
        a->queued_bytes += take;
        a->written_bytes += take;
        accepted += take;
    }
    a->submitted = a->written_bytes / a->block;
    return accepted;
}

static void refresh(bav_audio *a)
{
    MMTIME mt;
    if (a == NULL || !a->opened) { return; }
    reap(a);
    mt.wType = TIME_SAMPLES;
    if (waveOutGetPosition(a->h, &mt, sizeof mt) == MMSYSERR_NOERROR &&
        mt.wType == TIME_SAMPLES) {
        uint64_t pos = (uint64_t)mt.u.sample;
        a->presented = (pos >= a->start_pos) ? (pos - a->start_pos) : 0u;
    }
}

void bav_audio_status_get(bav_audio *a, bav_audio_status *st)
{
    if (st == NULL) { return; }
    memset(st, 0, sizeof *st);
    if (a == NULL || !a->opened) {
        st->t_audio = st->t_monotonic = bav_monotonic();
        return;
    }
    refresh(a);
    st->t_audio = (double)a->presented / (double)a->rate;
    st->t_monotonic = bav_monotonic() - a->epoch;
    st->submitted = a->submitted;
    st->presented = a->presented;
    st->queued = a->queued_bytes / a->block;
    st->underruns = a->underruns;
}

int bav_audio_drain(bav_audio *a, double timeout_seconds)
{
    double t0;
    if (a == NULL || !a->opened) { return 1; }
    t0 = bav_monotonic();
    for (;;) {
        int busy = 0, i;
        refresh(a);
        for (i = 0; i < BAV_AUDIO_CHUNKS; i++) { if (a->used[i]) { busy = 1; } }
        if (!busy) { return 1; }
        if (bav_monotonic() - t0 > timeout_seconds) { return 0; }
        Sleep(2);
    }
}

void bav_audio_close(bav_audio *a)
{
    int i;
    if (a == NULL) { return; }
    if (a->opened) {
        waveOutReset(a->h);
        for (i = 0; i < BAV_AUDIO_CHUNKS; i++) {
            if (a->used[i]) {
                waveOutUnprepareHeader(a->h, &a->hdr[i], sizeof(WAVEHDR));
                a->used[i] = 0;
            }
        }
        waveOutClose(a->h);
    }
    for (i = 0; i < BAV_AUDIO_CHUNKS; i++) { free(a->buf[i]); }
    free(a);
}
