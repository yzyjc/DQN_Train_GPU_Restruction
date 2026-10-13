// GPT6-Astra: source-level TUI integration, without a live Trainer connection.
#define UNICODE
#define _UNICODE
#define wWinMain unused_tui_entry
#include "../DQN_Main_TUI_No_Terminal/DQN_Main_TUI_No_Terminal.cpp"
#undef wWinMain
#undef printf
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    NEED(CONFIG_SLIDER_COUNT==10);
    reset_sliders_to_default();
    NEED(g_learning_rate_auto && !g_slider_dirty[1]);
    NEED(g_config_sliders[6].value==0 && g_config_sliders[7].value==.10f);
    NEED(g_config_sliders[9].value==.05f);
    memset(g_slider_dirty,0,sizeof(g_slider_dirty));
    parse_status_push("{\"white_draw_length_scale\":0.08}");NEED(g_config_sliders[9].value==.08f);
    parse_status_push("{\"step_outcome_enabled\":1,\"black_fast_win_scale\":0.12,\"white_survival_scale\":0.08}");
    NEED(g_config_sliders[6].value==1 && g_config_sliders[7].value==.12f);
    NEED(g_config_sliders[8].value==.08f);
    g_slider_dirty[7]=1;
    parse_status_push("{\"black_fast_win_scale\":0.01}");NEED(g_config_sliders[7].value==.12f);
    parse_status_push("{\"white_survival_scale\":9}");NEED(g_config_sliders[8].value==.08f);
    vt_init(200,80);g_active_tab=0;g_force_redraw=1;
    for(int width:{80,180})for(int focus=0;focus<(int)CONFIG_SLIDER_COUNT;++focus){
        g_config_focus=focus;render_config(0,0,width,12);
        NEED(g_config_sliders[focus].y>=CONTENT_Y+2 && g_config_sliders[focus].y<CONTENT_Y+11);
    }
    puts("PASS outcome TUI defaults, live readback, dirty preservation and range checks");return 0;
}
