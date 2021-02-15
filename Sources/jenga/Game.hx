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
            create(64);
            notifyOnUpdate(update);
        });
    }

    public function clear() {
        for( block in blocks ) {
            // var drag = block.getTrait( BlockDrag );
            // drag.remove();
            // var rb = block.getTrait( RigidBody );
            // rb.delete();
            // //PhysicsWorld.active.removeRigidBody( rb );
            block.remove();
        }
        // PhysicsWorld.active.reset();
        blocks = [];
    }

    public function create( numBlocks : Int ) {
        clear();
        var container = Scene.active.getEmpty('BlockContainer');
        function spawnBlock() {
            Scene.active.spawnObject( 'Block', container, obj -> {
                obj.name = 'Block_'+blocks.length;
                // obj.addTrait( new BlockDrag() );
                // trace("ADD DRAG");
                blocks.push( obj );
                if( blocks.length == numBlocks ) {
                    start();
                } else {
                    spawnBlock();
                }
            });
        }
        spawnBlock();
    }

    public function start() {

        /* 
        var blocks = this.blocks.slice( 0, 32 );
        for( i in 32...blocks.length ) {
            blocks[i].remove();
        }
        */

        /*
        _blocks = _blocks.slice( 0, 32 );
        for( i in 32...blocks.length ) {
            PhysicsWorld.active.removeRigidBody();
            blocks[i].remove();
        }
        */

        //var cam = Scene.active.camera;
        //cam.transform.rotate( Vec4.zAxis(), Math.PI/2 );
        camFocus.transform.rot.set(0,0,0,1);
        camFocus.transform.rotate( Vec4.zAxis(), Math.PI/4 );
        camFocus.transform.buildMatrix();

        var ix = 0, iz = 0;
        var rz = false;
        var px = - (blockX*2 + blockX/2), py = px;
        var randomOffsetFactor = 0.01;
        for( i in 0...blocks.length ) {
            var b = blocks[i];
            b.transform.loc.set(0,0,0);
            b.transform.rot = new iron.math.Quat();
            if( i != 0 && i % 4 == 0 ) {
                ix = 0;
                px = py = - (blockX*2 + blockX/2);
                rz = !rz;
                iz++;
            }
            if( rz ) {
                b.transform.rotate( Vec4.zAxis(), Math.PI/2 );
                py += blockX + Math.random() * randomOffsetFactor;
                b.transform.loc.y = py;
            } else {
                px += blockX + Math.random() * randomOffsetFactor;
                b.transform.loc.x = px;
                //b.transform.loc.x = (ix*blockX) - ((4*blockX)/2) + blockX/2;
            }
            ix++;
            b.transform.loc.z = iz * blockZ + (blockZ/2);
            b.transform.buildMatrix();
            var body = b.getTrait( RigidBody );
            body.syncTransform();
        }
    }

    function update() {
        if( blocks.length == 0 )
            return;
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
                //rotateStart.set( mouse.x, mouse.y );
                if( lastMouseX != 0 ) {
                    var moved = mouse.x-lastMouseX;
                    camFocus.transform.rotate(Vec4.zAxis(), moved/400 );
                }
                lastMouseX = mouse.x;
            } else {
                lastMouseX = 0;
            }
        }
    }
}