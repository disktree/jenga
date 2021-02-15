package jenga;

import zui.Id;
import zui.Themes;

class Menu extends iron.Trait {
    
    var game : Game;
    var ui : Zui;

    public function new() {
        super();
        notifyOnInit( () -> {
            game = Scene.active.getTrait( jenga.Game );
            Data.getFont( 'mono.ttf', font -> {
                ui = new Zui( { font : font } );
                notifyOnUpdate( update );
                notifyOnRender2D( render2D );
            });
        });
    }

    function update() {
        var kb = iron.system.Input.getKeyboard();
        if( kb.started( "escape" ) || kb.started( "space" ) ) {
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
        if( ui.window( hwin, 1, 1, 200, 200, false ) ) {
            if( ui.button( "JENGA!", Left ) ) game.start();
            //if( ui.button( "CLEAR", Left ) ) game.clear();
            if( ui.button( "CREATE", Left ) ) game.create(32);
        }
        ui.end();
        g.begin( false );
    }
}