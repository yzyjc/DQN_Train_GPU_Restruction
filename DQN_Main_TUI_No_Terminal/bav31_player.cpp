#include "bav31/bav.h"
#include "resource.h"
#include "terminal_window.h"
#include "virtual_terminal.h"

#include <windows.h>
#include <stddef.h>
#include <stdint.h>

static const char kBav31Ramp[] = " .:-=+*#%@";

static int bav31_poll_stop(void)
{
    int key;
    win_pump_messages();
    while ((key = vt_key_poll()) != -1) {
        if (key == 27 || key == 'q' || key == 'Q') return 1;
    }
    return !g_running;
}

static int bav31_submit_audio(bav_audio* audio, const uint8_t* pcm, size_t bytes)
{
    size_t sent = 0;
    while (sent < bytes) {
        size_t accepted = bav_audio_submit(audio, pcm + sent, bytes - sent);
        sent += accepted;
        if (sent < bytes) {
            if (bav31_poll_stop()) return 0;
            Sleep(1);
        }
    }
    return 1;
}

static void bav31_draw_frame(const bav_decoder* decoder)
{
    const uint8_t* pixels = bav_dec_pixels(decoder);
    const int screen_w = vt_get_width();
    const int screen_h = vt_get_height();
    const int width = (int)decoder->width;
    const int height = (int)decoder->height;
    int ox = (screen_w - width) / 2;
    int oy = (screen_h - height) / 2;
    char line[BAV_GRAY10_MAX_DIM + 1];

    if (ox < 0) ox = 0;
    if (oy < 0) oy = 0;
    vt_set_color(15, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t level = pixels[(size_t)y * (size_t)width + (size_t)x];
            line[x] = kBav31Ramp[level <= 9u ? level : 9u];
        }
        line[width] = '\0';
        vt_gotoxy(ox, oy + y);
        vt_printf("%s", line);
    }
    vt_present();
}

void play_bav(void)
{
    HMODULE module = GetModuleHandleW(NULL);
    HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(IDR_BADAPPLE_BAV31), RT_RCDATA);
    HGLOBAL loaded = resource ? LoadResource(module, resource) : NULL;
    const uint8_t* data = loaded ? (const uint8_t*)LockResource(loaded) : NULL;
    DWORD data_size = resource ? SizeofResource(module, resource) : 0;
    bav_decoder decoder = {};
    bav_audio* audio = NULL;
    int stopped = 0;
    int audio_period_set = 0;
    double mono_start = 0.0;

    vt_set_animation_enabled(0);
    vt_clear_screen();
    if (!data || data_size == 0 || bav_dec_open(&decoder, data, (size_t)data_size) != BAV_OK) {
        vt_set_color(12, 0);
        vt_gotoxy(2, 2);
        vt_printf("Bad Apple BAV 3.1 资源无法读取或校验失败。\n");
        vt_present();
        Sleep(1000);
        goto cleanup;
    }

    if (decoder.channels != 0u) audio = bav_audio_open(decoder.sample_rate, decoder.channels);
    if (audio) {
        timeBeginPeriod(1);
        audio_period_set = 1;
    } else {
        mono_start = bav_monotonic();
    }

    for (;;) {
        const uint8_t* pcm = NULL;
        size_t pcm_bytes = 0;
        const uint32_t frame = decoder.index;
        double target;

        if (bav31_poll_stop()) { stopped = 1; break; }
        if (decoder.truncated || frame >= decoder.frame_count) break;
        if (bav_dec_next(&decoder, &pcm, &pcm_bytes) == 0) break;
        if (audio && pcm_bytes && !bav31_submit_audio(audio, pcm, pcm_bytes)) {
            stopped = 1;
            break;
        }

        target = bav_dec_display_time(&decoder, frame);
        for (;;) {
            double now;
            double remain;
            bav_audio_status status;

            if (bav31_poll_stop()) { stopped = 1; break; }
            if (audio) {
                bav_audio_status_get(audio, &status);
                now = status.t_audio;
            } else {
                now = bav_monotonic() - mono_start;
            }
            if (now >= target - 0.005) break;
            remain = target - 0.005 - now;
            if (remain > 0.003) Sleep((DWORD)((remain - 0.001) * 1000.0));
            else Sleep(0);
        }
        if (stopped) break;
        bav31_draw_frame(&decoder);
    }

    if (audio && !stopped) bav_audio_drain(audio, 5.0);
    vt_clear_screen();
    vt_set_color(7, 0);
    vt_gotoxy(2, 2);
    vt_printf("Bad Apple BAV 3.1-r1 播放%s（帧 %u / %u，损坏单元 %u）\n",
        stopped ? "已跳过" : "结束", decoder.index, decoder.frame_count, decoder.errors);
    vt_present();
    Sleep(650);

cleanup:
    if (audio) bav_audio_close(audio);
    if (audio_period_set) timeEndPeriod(1);
    bav_dec_close(&decoder);
    vt_set_animation_enabled(1);
    vt_mark_all_dirty();
}
