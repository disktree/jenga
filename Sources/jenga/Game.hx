package jenga;

typedef Config = {
    var numBlocks : Int;
    var blockObject : String;
    //var ranPosFactor : Float;
}

class Game extends iron.Trait {

    static inline var ROW_SIZE = 4;

    public var blocks(default,null) : Array<Object>;
    public var blockDragged(default,null) : Object;
    public var blockDim(default,null)  : Vec4;

    var camRigZ : Object;
    var camRigX : Object;
    var mouse : Mouse;
    var lastMouseX = 0.0;
    var lastMouseY = 0.0;

    public function new() {
        super();
        trace( 'JENGA ${Main.projectVersion}' );
        notifyOnInit( () -> {
            #if kha_html5
            Scene.active.world.raw.probe.strength = 0.2; // HACK
            #end
            camRigZ = Scene.active.getEmpty('CameraRigZ');
            camRigX = Scene.active.getEmpty('CameraRigX');
            mouse = Input.getMouse();
            blocks = [];
            create({
                numBlocks: 72,
                blockObject : "Block3"
            }, start );
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

    public function create( config : Config, cb : Void->Void ) {
        //clear();
        var blockName = config.blockObject;
        var container = Scene.active.getEmpty('BlockContainer');
        function spawnBlock() {
            Scene.active.spawnObject( blockName, container, obj -> {
                obj.name = 'Block_'+blocks.length;
                blocks.push( obj );
                if( blocks.length == config.numBlocks ) {
                    blockDim = blocks[0].transform.dim;
                    cb();
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

        camRigZ.transform.rot.set(0,0,0,1);
        camRigZ.transform.rotate( Vec4.zAxis(), Math.PI/4 );
        camRigZ.transform.buildMatrix();

        var numRows = Std.int(blocks.length / ROW_SIZE);
        var ranPosFactor = 0.005 + Math.random() * 0.01;
        var ranRotFactor = 0.02 + Math.random() * 0.04;

        var ix = 0;
        var px = 0.0;
        var pz = blockDim.z/2;
        var rz = false;

        for( iz in 0...numRows ) {
            px = -(blockDim.x*2 + blockDim.x/2);
            for( ix in 0...ROW_SIZE ) {
                var b = blocks[(iz*ROW_SIZE)+ix];
                b.transform.loc.set(0,0,0);
                b.transform.rot = new Quat();
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
            pz += blockDim.z + 0.01;
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

        var rotZ = 0.0;
        var rotX = 0.0;

        if( blockDragged != null ) {
            //..
        } else {
            if( mouse.down("right")) {
                if( lastMouseX != 0 ) {
                    var moved = mouse.x - lastMouseX;
                    if( moved != 0 ) {
                        rotZ = moved / 200;
                    }
                }
                if( lastMouseY != 0 ) {
                    var moved = mouse.y - lastMouseY;
                    if( moved != 0 ) {
                        rotX = moved / 2000;
                    }
                }
                lastMouseX = mouse.x;
                lastMouseY = mouse.y;

            } else {
                lastMouseX = 0;
                lastMouseY = 0;
            }
        }

        camRigZ.transform.rotate( new Vec4( 0,0,rotZ,1), 1 );
        
        var rx = camRigX.transform.rot.x + rotX;
        if( rx > 0.3 ) rx = 0.3;
        else if( rx < -0.3 ) rx = -0.3;
        camRigX.transform.rot.x = rx;

        camRigX.transform.buildMatrix();
    }
}
