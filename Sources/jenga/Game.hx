package jenga;

class Game extends iron.Trait {
    
    static inline var blockX = 0.41;
    static inline var blockY = 2.0;
    static inline var blockZ = 0.305;
    static inline var numBlocks = 64;

    var blocks : Array<Object>;

    var mouse : Mouse;
    var camFocus : Object;
    var lastMouseX = 0.0;
    var lastMouseY = 0.0;

    var blockDragged : Object;

    public function new() {
        super();
        notifyOnInit( () -> {
            #if kha_html5
            js.Browser.window.oncontextmenu = e -> e.preventDefault();
            #end
            camFocus = Scene.active.getEmpty('Focus');
            mouse = Input.getMouse();
            blocks = [];
            function spawnBlock() {
                Scene.active.spawnObject( 'Block', null, obj -> {
                    obj.name = 'Block_'+blocks.length;
                    blocks.push( obj );
                    /*
                    var i = blocks.length;
                   Event.add('drag', () -> {
                        trace("DRAG "+obj.name );
                    }, i ); */
                    if( blocks.length == numBlocks ) {
                        start();
                    } else {
                        spawnBlock();
                    }
                });
            }
           spawnBlock();
           notifyOnUpdate(update);
        });
    }

    public function start( ) {

        //var cam = Scene.active.camera;
        //trace(cam);
        //cam.transform.rotate( Vec4.zAxis(), Math.PI/2 );
        camFocus.transform.rot.set(0,0,0,1);
        camFocus.transform.rotate( Vec4.zAxis(), Math.PI/4 );
        camFocus.transform.buildMatrix();

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
        blockDragged = null;
        for( block in blocks ) {
            var drag = block.getTrait(BlockDrag);
            if( drag.pickedBody != null ) {
                blockDragged = drag.pickedBody.object;
                //trace(drag.pickedBody.object.name);

            }
        }
        if( blockDragged != null ) {
            //..

        } else {
            if( mouse.down("right")) {
                //trace("RORARTWE START");
                //rotateStart.set( mouse.x, mouse.y );
                if( lastMouseX != 0 ) {
                    var moved = mouse.x-lastMouseX;
                    camFocus.transform.rotate(Vec4.zAxis(), moved/200 );
                }
                lastMouseX = mouse.x;
                /* if( lastMouseY != 0 ) {
                    var moved = mouse.y-lastMouseY;
                    camFocus.transform.rotate(Vec4.yAxis(), moved/1000 );
                }
                lastMouseY = mouse.y; */
            } else {
                lastMouseX = 0;
            }
        }
    }
}