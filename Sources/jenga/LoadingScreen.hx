package jenga;

class LoadingScreen {

	static inline var BAR_W = 200;
	static inline var BAR_H = 20;

	public static function render( g : kha.graphics2.Graphics, assetsLoaded : Int, assetsTotal : Int ) {
		trace( 'Loading asset $assetsLoaded/$assetsTotal' );
		final sw = System.windowWidth();
		final sh = System.windowHeight();
		final cx = Std.int( sw / 2 );
		final cy = Std.int( sh / 2 );
		final hw = Std.int( BAR_W / 2 );
		g.color = 0xff000000;
		g.fillRect( 0, 0, sw, sh );
		g.color = 0xffffffff;
		g.fillRect( cx - hw, cy - hw, BAR_W / assetsTotal * assetsLoaded, BAR_H );
		g.color = 0xffffffff;
	}

}
