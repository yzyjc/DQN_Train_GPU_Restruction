# 统一控件

包含 `ui_controls.h`，在页面绘制时调用。业务启停、命令发送仍由原事件处理负责；绘制不会操作进程。

```cpp
ui_button(x, y, "CONSOLE", {running, false}, ui_lamp_button());
ui_button(x, y, "记忆开关", {enabled, false}, ui_toggle_button());
ui_button(x, y, "应用全部", {false, pressed}, ui_push_button());
```

- 指示灯开关：关闭时空心圆、灰字；开启时青灯、白字。
- 文字开关：关闭时灰字，开启时青字。
- 普通按钮：按下时青字，松开时灰字；焦点不会使它持续激活。
- 文案可以每次变化。用 `ui_button_width` 布局，返回的矩形 `contains` 判断点击；两者使用相同宽度算法。
- 可配置 `indicator`、`toggle`、`padding` 和颜色。`indicator=false` 去掉灯；`toggle=false` 按 `pressed` 而非 `active` 绘制。
- 页面负责清除旧文案覆盖区域。控件仅绘制一行，使用当前终端字体，不单独改变字号。

```cpp
ui_number(x, y, "123.45");        // 指定位置，机械滚轮
ui_number(x, y, "123.45", false); // 静态数字
ui_number_printf("Episode: %d", episode); // 当前光标位置，格式化输出
```

数字入口复用 `rolling_digit.h` 与虚拟终端逐帧渲染。重复提交相同数字不会重启动画，新数字可打断当前滚动；标点和文字不滚动。数字状态按终端格子保存，切页或缩放沿用终端重置机制，不跨页面继承动画。页面只提交目标，终端负责后续帧。
