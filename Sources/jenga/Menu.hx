package jenga;

import zui.Id;
import zui.Themes;

class Menu extends iron.Trait {
    
    var ui : Zui;
    var mouse : Mouse;
    var game : Game;
    var opacity = 1.0;

    public function new() {
        super();
        notifyOnInit( () -> {
            mouse = Input.getMouse();
            UI.init( () -> {
                ui = UI.create();
                game = Scene.active.getTrait( jenga.Game );
                notifyOnUpdate( update );
                notifyOnRender2D( render2D );
                Tween.to({
                    target: this,
                    props: { opacity: 0.0 },
                    duration: 1.0,
                    delay: 1.0,
                    //done: () -> removeRender2D( render2D )
                });
            });
        });
    }

    function update() {
        var kb = iron.system.Input.getKeyboard();
        if( kb.started( "escape" ) || kb.started( "space" ) || kb.started( "f5" ) ) {
            game.start();
            game.resetCamera();
        }
    }

    function render2D( g : kha.graphics2.Graphics ) {
        
        if( ui == null ) return;
        
        var sw = System.windowWidth(), sh = System.windowHeight();
        
        g.end();

		ui.begin( g );
		//ui.beginRegion( g );
        g.opacity = opacity;
        var hwin = Id.handle();
        hwin.redraws = 1;
        if( ui.window( hwin, 1, 256, sw, 512, false ) ) {
            //ui.ops.theme.TEXT_COL = 0xff0000ff;
            ui.button('JENGA',Center);
        }
        ui.end();

        /* g.color = 0xffffffff;
		g.font = UI.fontTitle;
		g.fontSize = Std.int(256);
        var text = 'JENGA';
        var textWidth = UI.fontTitle.width( g.fontSize, text );
        g.drawString( text, sw/2 - textWidth/2, sh/2 - g.fontSize/2 );  */
       
        g.color = 0xff000000;
		g.font = UI.fontRegular;
		g.fontSize = Std.int(12);
        var text = 'v'+Main.projectVersion;
        var textWidth = UI.fontRegular.width( g.fontSize, text );
        g.drawString( text, sw-(textWidth + (g.fontSize*1.1)), sh-(g.fontSize*1.5) ); 

        g.begin( false );
    }
}