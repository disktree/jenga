package jenga;

class Game extends iron.Trait {
    
    static inline var blockX = 0.4;
    static inline var blockY = 2.0;
    static inline var blockZ = 0.3;
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
            Scene.active.world.raw.probe.strength = 0.4; // HACK
            #end
            camFocus = Scene.active.getEmpty('Focus');
            mouse = Input.getMouse();
            blocks = [];
            create(68);
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
            blocks[i].remove()
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

        var rows = Std.int(blocks.length/4);
        var ranPosFactor = 0.005 + Math.random() * 0.01;
        var ranRotFactor = 0.02 + Math.random()*0.04;

        var ix = 0;
        var px = 0.0;
        var pz = blockZ/2;
        var rz = false;

        for( iz in 0...rows ) {
            px = -(blockX*2 + blockX/2);
            for( ix in 0...4 ) {
                var b = blocks[(iz*4)+ix];
                b.transform.loc.set(0,0,0);
                b.transform.rot = new iron.math.Quat();
                var ranRot = - ranRotFactor + Math.random()*(ranRotFactor*2);
                var ranPos = - ranPosFactor + Math.random()*(ranPosFactor*2);
                if( rz ) {
                    b.transform.rotate( Vec4.zAxis(), Math.PI/2 + ranRot );
                    px += blockX + ranPos;
                    b.transform.loc.y = px;
                } else {
                    b.transform.rotate( Vec4.zAxis(), ranRot );
                    px += blockX + ranPos;
                    b.transform.loc.x = px;
                }
                b.transform.loc.z = pz;
                b.transform.buildMatrix();
                var body = b.getTrait( RigidBody );
                body.syncTransform();
            }
            pz += blockZ+0.001;
            rz = !rz;
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