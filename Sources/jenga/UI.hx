package jenga;

import kha.Font;
import zui.Id;
import zui.Themes;
import zui.Zui;

class UI {

    public static var THEME : TTheme = {
		NAME: "Jenga.1",
		ACCENT_COL: 0xff000000,
		ACCENT_HOVER_COL: 0xfff0f0f0,
		ACCENT_SELECT_COL: 0xffffffff,
		ARROW_SIZE: 64,
		BUTTON_H: 40,
		BUTTON_COL: 0x00ffffff,
		BUTTON_HOVER_COL: 0x00ffffff,
		BUTTON_PRESSED_COL: 0x00ffffff,
		BUTTON_TEXT_COL: 0xffffffff,
		CHECK_SELECT_SIZE: 7,
		CHECK_SIZE: 12,
		CONTEXT_COL: 0xff222222,
		ELEMENT_H: 128,
		ELEMENT_OFFSET: 4,
		ELEMENT_W: 200,
		FILL_ACCENT_BG: false,
		FILL_BUTTON_BG: false,
		FILL_WINDOW_BG: false,
		FONT_SIZE: 256,
		HIGHLIGHT_COL: 0xff205d9c,
		LABEL_COL: 0xffffffff,
		LINK_STYLE: Line,
		PANEL_BG_COL: 0xff000000,
		SCROLL_W: 6,
		SEPARATOR_COL: 0xff0000ff,
		TAB_W: 6,
		TEXT_COL: 0xffffffff,
		TEXT_OFFSET: 8,
		WINDOW_BG_COL: 0x00000000,
		WINDOW_TINT_COL: 0xffffffff,
	};
    
	public static var THEME_2 : TTheme = {
		NAME: "Jenga.2",
		ACCENT_COL: 0xff000000,
		ACCENT_HOVER_COL: 0xfff0f0f0,
		ACCENT_SELECT_COL: 0xffffffff,
		ARROW_SIZE: 64,
		BUTTON_H: 40,
		BUTTON_COL: 0x00ffffff,
		BUTTON_HOVER_COL: 0x00ffffff,
		BUTTON_PRESSED_COL: 0x00ffffff,
		BUTTON_TEXT_COL: 0xffffffff,
		CHECK_SELECT_SIZE: 7,
		CHECK_SIZE: 12,
		CONTEXT_COL: 0xff222222,
		ELEMENT_H: 128,
		ELEMENT_OFFSET: 4,
		ELEMENT_W: 200,
		FILL_ACCENT_BG: false,
		FILL_BUTTON_BG: false,
		FILL_WINDOW_BG: false,
		FONT_SIZE: 256,
		HIGHLIGHT_COL: 0xff205d9c,
		LABEL_COL: 0xffffffff,
		LINK_STYLE: Line,
		PANEL_BG_COL: 0xff000000,
		SCROLL_W: 6,
		SEPARATOR_COL: 0xff0000ff,
		TAB_W: 6,
		TEXT_COL: 0xffffffff,
		TEXT_OFFSET: 8,
		WINDOW_BG_COL: 0x00000000,
		WINDOW_TINT_COL: 0xffffffff,
	};

	public static var fontTitle(default,null) : kha.Font;
	public static var fontRegular(default,null) : kha.Font;

	//public static var theme : Map<String,Theme>;

    public static inline function init( ?done : Void->Void ) {
		Data.getFont( 'AbrilFatface-Regular.ttf', f -> {
			fontTitle = f;
			Data.getFont( 'mono.ttf', f -> {
				fontRegular = f;
				if( done != null ) done();
			});
		});
	}

    public static function create( ?font : Font, ?theme : TTheme ) : Zui {
		if( font == null ) font = fontTitle;
		if( theme == null ) theme = THEME;
        return new Zui( { font : font, theme : theme } );
    }

}