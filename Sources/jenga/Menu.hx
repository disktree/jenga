package jenga;

import zui.Id;
import zui.Themes;

class Menu extends iron.Trait {
    
    var ui : Zui;
    var game : Game;

    public function new() {
        super();
        notifyOnInit( () -> {
            UI.init( () -> {
                ui = UI.create();
                game = Scene.active.getTrait( jenga.Game );
                notifyOnUpdate( update );
                notifyOnRender2D( render2D );
            });
        });
    }

    function update() {
        var kb = iron.system.Input.getKeyboard();
        if( kb.started( "escape" ) || kb.started( "space" ) || kb.started( "f5" ) ) {
            game.start();
        }
    }

    function render2D( g : kha.graphics2.Graphics ) {
        
        if( ui == null ) return;
        
        var sw = System.windowWidth(), sh = System.windowHeight();
        
        g.end();

		ui.begin( g );
		//ui.beginRegion( g );
        g.opacity = 1;
        var hwin = Id.handle();
        hwin.redraws = 1;
        if( ui.window( hwin, 1, 1, 120, 40, false ) ) {
            if( ui.button( "JENGA!", Left ) ) game.start();
            //if( ui.button( "CLEAR", Left ) ) game.clear();
            //if( ui.button( "CREATE", Left ) ) game.create(32);
        }
        ui.end();

        g.color = 0xff000000;
		g.font = UI.font;
		g.fontSize = Std.int(12);
        var text = 'v'+Main.projectVersion;
        var textWidth = UI.font.width( g.fontSize, text );
        g.drawString( text, sw-(textWidth + (g.fontSize*1.1)), sh-(g.fontSize*1.5) ); 

        g.begin( false );
    }
}