package jenga;

class Game extends iron.Trait {

    public var blocks(default,null) : Array<Object>;
    public var blockDragged(default,null) : Object;
    public var blockDim(default,null)  : Vec4;

    var camFocus : Object;
    var mouse : Mouse;
    var lastMouseX = 0.0;
    var lastMouseY = 0.0;

    public function new() {
        super();
        notifyOnInit( () -> {
            #if kha_html5
            Scene.active.world.raw.probe.strength = 0.3; // HACK
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
        var blockName = "Block2";
        var container = Scene.active.getEmpty('BlockContainer');
        function spawnBlock() {
            Scene.active.spawnObject( blockName, container, obj -> {
                if( blockDim == null ) {
                    blockDim = obj.transform.dim;
                }
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

        for( block in blocks ) {
            var body = block.getTrait(RigidBody);
            body.setAngularVelocity(0,0,0);
            body.setLinearVelocity(0,0,0);
        }

        camFocus.transform.rot.set(0,0,0,1);
        camFocus.transform.rotate( Vec4.zAxis(), Math.PI/4 );
        camFocus.transform.buildMatrix();

        var rows = Std.int(blocks.length/4);
        var ranPosFactor = 0.005 + Math.random() * 0.01;
        var ranRotFactor = 0.02 + Math.random()*0.04;

        var ix = 0;
        var px = 0.0;
        var pz = blockDim.z/2;
        var rz = false;

        for( iz in 0...rows ) {
            px = -(blockDim.x*2 + blockDim.x/2);
            for( ix in 0...4 ) {
                var b = blocks[(iz*4)+ix];
                b.transform.loc.set(0,0,0);
                b.transform.rot = new iron.math.Quat();
                var ranRot = - ranRotFactor + Math.random()*(ranRotFactor*2);
                var ranPos = - ranPosFactor + Math.random()*(ranPosFactor*2);
                if( rz ) {
                    b.transform.rotate( Vec4.zAxis(), Math.PI/2 + ranRot );
                    px += blockDim.x + ranPos;
                    b.transform.loc.y = px;
                } else {
                    b.transform.rotate( Vec4.zAxis(), ranRot );
                    px += blockDim.x + ranPos;
                    b.transform.loc.x = px;
                }
                b.transform.loc.z = pz;
                b.transform.buildMatrix();
                var body = b.getTrait( RigidBody );
                body.syncTransform();
            }
            pz += blockDim.z + 0.001;
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