/* BAV 3.1-r1 — reference implementation core.
 *
 * Spec: BAV31-r1-规范.md (frozen).  This header deliberately contains no
 * platform or I/O code: the same core serves the encoder tool, the probe and
 * the in-memory player that gets compiled into the host executable (the final
 * delivered artifact is a .h holding the whole file as a const array).
 */
#ifndef BAV_H
#define BAV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- container constants (spec §2) ---------------- */
#define BAV_MAGIC             "BAV3"
#define BAV_VERSION_MAJOR     3
#define BAV_VERSION_MINOR     1
#define BAV_HEADER_SIZE       60
#define BAV_UNIT_SYNC         "BAVF"
#define BAV_UNIT_HEADER_SIZE  20

#define BAV_FOOTER_MAGIC      0x45564142u /* 'B','A','V','E' little-endian */
#define BAV_CHUNK_SEEK        0x4B454553u /* 'S','E','E','K' little-endian */

/* video_format (spec §2.2) */
#define BAV_VF_GRAY10         0x0000u

/* audio_sample_format (spec §2.2) */
#define BAV_ASF_S16LE         0x00u

/* unit_flags (spec §2.3) */
#define BAV_UF_KEYFRAME       0x01u
#define BAV_UF_NO_VIDEO_UPDATE 0x02u

/* GRAY10 frame_flags (spec §3.2) */
#define BAV_FF_IS_DIFF        0x01u
#define BAV_FF_HAS_RECT       0x02u

#define BAV_GRAY10_MAX_DIM    256

/* ---------------- growable byte buffer ---------------- */
typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
    int      oom;
} bav_buf;

void bav_buf_init(bav_buf *b);
void bav_buf_free(bav_buf *b);
int  bav_buf_put(bav_buf *b, const void *p, size_t n);
int  bav_buf_u8 (bav_buf *b, uint8_t  v);
int  bav_buf_u16(bav_buf *b, uint16_t v);
int  bav_buf_u32(bav_buf *b, uint32_t v);
int  bav_buf_u64(bav_buf *b, uint64_t v);

/* ---------------- checksum (spec §0.7) ----------------
 * CRC-32/ISO-HDLC.  Incremental form is required because a unit's CRC covers
 * "16 header bytes + video payload + audio payload" — three separate spans.
 * Self-test: bav_crc32("123456789", 9) == 0xCBF43926.
 */
uint32_t bav_crc32_init(void);
uint32_t bav_crc32_update(uint32_t state, const void *data, size_t len);
uint32_t bav_crc32_final(uint32_t state);
uint32_t bav_crc32(const void *data, size_t len);

/* ---------------- timing, integer-only (spec §4.2 / §2.7) ----------------
 * A_n  = floor(n * fs * den / num)      — sample boundary of frame n
 * gap  = max(1, floor(2 * num / den))   — max keyframe distance in frames
 * Both use the overflow-safe decomposition; see spec §4.2.
 */
uint64_t bav_a_n(uint64_t n, uint32_t fs, uint32_t num, uint32_t den);
uint64_t bav_frame_gap_max(uint32_t num, uint32_t den);
uint64_t bav_total_sample_frames(uint32_t frame_count, uint32_t fs,
                                 uint32_t num, uint32_t den);

/* ---------------- writer ---------------- */
typedef struct {
    uint32_t width, height;
    uint32_t fps_num, fps_den;
    uint32_t frame_count;
    uint32_t video_format;        /* BAV_VF_* */
    uint32_t audio_channels;      /* 0 = no audio, else 1..2 */
    uint32_t audio_sample_format; /* BAV_ASF_S16LE */
    uint32_t bytes_per_sample;    /* 0 when no audio, else 2 */
    uint32_t audio_sample_rate;   /* 0 when no audio, else 44100 */
    uint32_t max_frame_unit_bytes;/* 0 = unspecified */
} bav_params;

/* Appends a complete 60-byte FileHeader (total_sample_frames is derived). */
int bav_write_header(bav_buf *out, const bav_params *p);

/* Appends one FrameUnit: 20-byte header (with CRC) + video + audio payloads. */
int bav_write_unit(bav_buf *out, uint8_t unit_flags,
                   const uint8_t *video, size_t video_len,
                   const uint8_t *audio, size_t audio_len);

/* ---------------- GRAY10 encoder (spec §3) ----------------
 * Appends one GRAY10 video payload for `cur` against `prev`.  When the frame is
 * identical to `prev` and this is not a keyframe, *changed is set to 0 and
 * nothing is appended (the caller emits a NO_VIDEO_UPDATE unit instead).
 * Returns 0 only on a real failure (allocation, or a frame that needs more runs
 * than run_count:u16 can express, i.e. w*h > 65535).
 */
int bav_g10_encode(bav_buf *out, const uint8_t *cur, const uint8_t *prev,
                   uint32_t w, uint32_t h, int keyframe, int *changed,
                   uint32_t *run_count_out);

/* ---------------- decoder (spec §5) ----------------
 * Pure and platform-free: it never touches files, the console or audio.  Feed
 * it a memory buffer (a .bav read from disk, or one sealed into the image) and
 * it hands back one frame at a time plus the PCM slice that belongs to it.
 */
typedef struct {
    const uint8_t *data;
    size_t         len;
    size_t         off;                 /* offset of the next unit */
    uint32_t       width, height;
    uint32_t       fps_num, fps_den;
    uint32_t       frame_count;
    uint32_t       channels;            /* 0 = video only */
    uint32_t       sample_rate;
    uint32_t       bytes_per_sample;
    uint64_t       total_sample_frames;
    uint32_t       index;               /* next frame to decode */
    uint8_t       *fb;                  /* width*height GRAY10 values */
    uint32_t       errors;              /* damaged units skipped */
    uint32_t       resyncs;
    uint32_t       lost_frames;         /* frames that could not be recovered */
    uint64_t       pending_silence;     /* sample frames of silence to conceal
                                         * the lost frames (spec §5.3) */
    int            truncated;           /* stream ended before frame_count */
} bav_decoder;

#define BAV_OK           0
#define BAV_ERR_SHORT   -1
#define BAV_ERR_MAGIC   -2
#define BAV_ERR_VERSION -3
#define BAV_ERR_HEADER  -4
#define BAV_ERR_CRC     -5
#define BAV_ERR_FORMAT  -6
#define BAV_ERR_MEM     -7

const char *bav_strerror(int code);
int  bav_dec_open(bav_decoder *d, const void *data, size_t len);
/* Decodes d->index into d->fb; *audio/*audio_len receive that frame's PCM slice.
 * Returns 1 = a frame was produced, 0 = end of stream (check d->truncated). */
int  bav_dec_next(bav_decoder *d, const uint8_t **audio, size_t *audio_len);
const uint8_t *bav_dec_pixels(const bav_decoder *d);
uint64_t bav_dec_frame_samples(const bav_decoder *d, uint32_t n);
double   bav_dec_frame_start(const bav_decoder *d, uint32_t n);
double   bav_dec_display_time(const bav_decoder *d, uint32_t n);
void bav_dec_close(bav_decoder *d);

/* ---------------- audio output + the four clocks (spec §4.4) ----------------
 * Windows only.  waveOut goes through the shared audio engine, so it mixes
 * with whatever else is playing; waveOutGetPosition reports the driver's
 * playback position, i.e. the closest thing to "presented to the output".
 */
typedef struct bav_audio bav_audio;

typedef struct {
    double   t_audio;        /* T_a: (device position)/rate       - the master */
    double   t_monotonic;    /* fallback: monotonic clock         - for comparison */
    uint64_t submitted;      /* sample frames handed to the API   - write cursor level */
    uint64_t presented;      /* sample frames reported played     - DAC level */
    uint64_t queued;         /* sample frames still in our chunks  - queue depth */
    uint32_t underruns;
} bav_audio_status;

bav_audio *bav_audio_open(uint32_t sample_rate, uint32_t channels);
/* Copies PCM into the device queue.  Returns bytes accepted (may be 0 when the
 * queue is full: the caller should sleep briefly and retry). */
size_t bav_audio_submit(bav_audio *a, const void *pcm, size_t bytes);
void   bav_audio_status_get(bav_audio *a, bav_audio_status *st);
/* Blocks until everything queued has been played (spec §4.8 tail handling). */
int    bav_audio_drain(bav_audio *a, double timeout_seconds);
void   bav_audio_close(bav_audio *a);

/* Monotonic seconds; enough precision for A/V sync on any platform. */
double bav_monotonic(void);

/* ---------------- general encoder engine (spec §2/§3/§4) ----------------
 * The engine is source-agnostic: bav_make feeds it from files, the drag-and-
 * drop tool feeds it from an ffmpeg pipe.  Everything the frozen spec requires
 * (first-frame keyframe, §2.7 keyframe interval, NO_VIDEO_UPDATE, exact audio
 * payload length, consistent total_sample_frames) is enforced here.
 */
typedef struct {
    /* Fill exactly width*height bytes of GRAY10 values (0..9) for frame_index.
     * Return 0 on failure. */
    int (*read_frame)(void *ctx, uint32_t frame_index, uint8_t *out_gray);
    /* Fill exactly `bytes` of interleaved S16LE starting at sample frame
     * `sample_frame`.  Called with a monotonically increasing sample_frame, so
     * sources may stream.  Return 0 on failure. */
    int (*read_audio)(void *ctx, uint64_t sample_frame, uint8_t *out, size_t bytes);
    /* Called once per frame; return 0 to cancel.  May be NULL. */
    int (*progress)(void *ctx, uint32_t frame, uint32_t total);
    /* Optional: the final GRAY10 frame the engine actually encoded (after the
     * 8-bit map, level quantisation and hysteresis).  Valid only during the
     * call.  Used for verification dumps. */
    void (*on_frame_ready)(void *ctx, uint32_t frame_index, const uint8_t *eff);
    void *ctx;
} bav_source;

typedef struct {
    /* inputs */
    uint32_t width, height;
    uint32_t fps_num, fps_den;
    uint32_t frame_count;
    uint32_t channels;            /* 0 = video only */
    uint32_t sample_rate;
    uint32_t bytes_per_sample;
    uint32_t levels;              /* 10 = keep all ten levels */
    uint32_t hysteresis;          /* 0 = off; see docs */
    double   gamma;               /* 1.0 = linear 8-bit -> GRAY10 map */
    /* outputs (filled by bav_pack) */
    uint64_t video_bytes;
    uint64_t audio_bytes;
    uint64_t keyframes;
    uint64_t delta_frames;
    uint64_t no_update_frames;
    uint32_t max_unit_bytes;
    uint32_t max_runs;
} bav_pack_opts;

#define BAV_PACK_OK        0
#define BAV_PACK_ERR_IN   -1     /* a source callback failed */
#define BAV_PACK_ERR_MEM  -2
#define BAV_PACK_ERR_CANCEL -3
#define BAV_PACK_ERR_PARAM -4

/* Encodes the whole stream into `out` (complete file, header included). */
int bav_pack(bav_buf *out, bav_pack_opts *o, const bav_source *src);

/* ---------------- .h emission (the final deliverable format) ---------------- */
#define BAV_HDR_HEX     0        /* 0x2C,0x41,...      ~6.3 chars/byte */
#define BAV_HDR_OCT     1        /* "\054\101"          4.2 chars/byte */
#define BAV_HDR_AUTO    2        /* printable + octal   3.5 chars/byte */
#define BAV_HDR_BASE64  3        /* base64 + decoder    1.4 chars/byte */

const char *bav_hdr_encoding_name(int encoding);
int bav_hdr_parse_encoding(const char *name);
/* Writes a self-contained C header carrying `data` (a whole .bav) as a
 * compile-time constant.  `source_label` only appears in the generated comment.
 * Returns 0 on success. */
int bav_hdr_write_file(const void *data, size_t len,
                       const char *source_label, const char *out_path,
                       const char *symbol, int encoding);

#ifdef __cplusplus
}
#endif

#endif /* BAV_H */
