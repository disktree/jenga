package jenga;

import armory.trait.physics.PhysicsWorld;
import armory.trait.physics.RigidBody;
import iron.Scene;
import iron.math.Vec4;
import iron.object.Object;
import iron.system.Input;

class Game extends iron.Trait {
    
    static inline var blockX = 0.41;
    static inline var blockY = 2.0;
    static inline var blockZ = 0.305;
    static inline var numBlocks = 64;

    var blocks : Array<Object>;

    var mouse : Mouse;
    var camFocus : Object;
    var lastMouseX = 0.0;

    public function new() {
        super();
        notifyOnInit( () -> {
            blocks = [];
            function spawnBlock() {
                Scene.active.spawnObject( 'Block', null, obj -> {
                    blocks.push( obj );
                    if( blocks.length == numBlocks ) {
                        start();
                    } else {
                        spawnBlock();
                    }
                });
            }
           spawnBlock();
           #if kha_html5
           js.Browser.window.oncontextmenu = e -> e.preventDefault();
           #end
           camFocus = Scene.active.getEmpty('Focus');
           mouse = Input.getMouse();
           notifyOnUpdate(update);
        });
    }

    public function start( ) {
        var ix = 0;
        var iz = 0;
        var rz = false;
        for( i in 0...blocks.length ) {
            var b = blocks[i];
            b.transform.loc.set(0,0,0);
            b.transform.rot = new iron.math.Quat();
            if( i != 0 && i % 4 == 0 ) {
                ix = 0;
                rz = !rz;
                iz++;
            }
            if( rz ) {
                b.transform.rotate( Vec4.zAxis(), Math.PI/2 );
                b.transform.loc.y = (ix*blockX) - ((4*blockX)/2) + blockX/2;
            } else {
                b.transform.loc.x = (ix*blockX) - ((4*blockX)/2) + blockX/2;
            }
            ix++;
            b.transform.loc.z = iz * blockZ + (blockZ/2);
            b.transform.buildMatrix();
            var body = b.getTrait( RigidBody );
            body.syncTransform();
        }
    }

    function update() {
        if( mouse.down("right")) {
            //trace("RORARTWE START");
            //rotateStart.set( mouse.x, mouse.y );
            if( lastMouseX != 0 ) {
                var moved = mouse.x-lastMouseX;
                camFocus.transform.rotate(Vec4.zAxis(), moved/200 );
            }
            lastMouseX = mouse.x;
        } else {
            lastMouseX = 0;
        }
    }
}