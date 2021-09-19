package jenga;

typedef Config = {
    var numBlocks : Int;
    var blockObject : String;
    //var ranPosFactor : Float;
}

class Game extends iron.Trait {

    static inline var ROW_SIZE = 4;
    static inline var ZOOM_SPEED = 0.4;
    static inline var ZOOM_DEFAULT = -6.0;
    static inline var ZOOM_MIN = -4.0;
    static inline var ZOOM_MAX = -8.0;
    static inline var ROT_X_MIN = -0.3;
    static inline var ROT_X_MAX = 0.2;

    public var blockDim(default,null)  : Vec4;
    public var blocks(default,null) : Array<Object>;
    public var blockDragged(default,null) : Object;

    var lastBlockDragged : Object;
    var cam : CameraObject;
    var camRigZ : Object;
    var camRigX : Object;
    var lastMouseX = 0.0;
    var lastMouseY = 0.0;
    var mouse : Mouse;
    var keyboard : Keyboard;

    public function new() {
        super();
        trace( 'JENGA ${Main.projectVersion}' );
        notifyOnInit( () -> {
            #if kha_html5
            Scene.active.world.raw.probe.strength = 0.2; // HACK
            #end
            cam = Scene.active.camera;
            camRigZ = Scene.active.getEmpty('CameraRigZ');
            camRigX = Scene.active.getEmpty('CameraRigX');
            mouse = Input.getMouse();
            keyboard = Input.getKeyboard();
            blocks = [];
            create({
                numBlocks: 64,
                blockObject : "Block4"
            }, () -> {
                start();
                resetCamera( 0.5 );
            });
            notifyOnUpdate(update);
            BlockDrag.onDragStart = b  -> {
                trace("Block drag start: "+b.name);
                blockDragged = b;
                lastBlockDragged = null;
            }
            BlockDrag.onDragEnd = b  -> {
                trace("Block drag end: "+b.name);
                lastBlockDragged = blockDragged;
                blockDragged = null;
            }
        });
    }

    public function create( config : Config, cb : Void->Void ) {
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

        var numRows = Std.int(blocks.length / ROW_SIZE);
        var ranPosFactor = 0.005 + Math.random() * 0.01;
        var ranRotFactor = 0.02 + Math.random() * 0.04;
        var zSpace = 0.01;

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
            pz += blockDim.z + zSpace;
            rz = !rz;
        }

        /* for( i in 0...ROW_SIZE ) {
            blocks[Std.int(numRows * ROW_SIZE) - (i+1)].getTrait(BlockDrag).enabled = false;
        } */

        //resetCamera();
    }

    function update() {

        /*
        // Detect physics contacts

        if( blockDragged != null ) {
            var body = blockDragged.getTrait(RigidBody);
            var contacts = PhysicsWorld.active.getContactPairs( body );
			if( contacts != null ) {
				trace(contacts.length+' contacts');
            }
        } else if( lastBlockDragged != null ) {
            var body = lastBlockDragged.getTrait(RigidBody);
            var contacts = PhysicsWorld.active.getContactPairs( body );
			if( contacts != null ) {
				trace(contacts.length+' contacts');
                for( contact in contacts ) {
                    var rb = PhysicsWorld.active.rbMap.get( contact.a );
                    if( rb == body ) rb = PhysicsWorld.active.rbMap.get( contact.b );
                    //trace(rb.object.name.);
                    if( rb.object.name == 'Ground' ) {
                        trace("HIT FLOOR");
                        var t = lastBlockDragged.getTrait( BlockDrag );
                        trace(t!=null);
                        lastBlockDragged.removeTrait(t);
                        //lastBlockDragged.remove();
                        lastBlockDragged = null;
                        return;
                    }
                }

            }
        }
        */

        // Determine if blocks is below default z pos

        /* var numRows = 16;
        var pz = 0.0;
        for( iz in 1...numRows ) {
            pz = iz * blockDim.z;
            for( ix in 0...ROW_SIZE ) {
                var block = blocks[(iz*4)+(ix)];
                if( block.transform.loc.z < pz ) {
                    if( block != blockDragged )
                        trace(block.name+' is dead');
                }
            }
        } */
       
        // Camera controls

        if( keyboard.started('0') ) {
            resetCamera();
            return;
        }

        var rotZ = 0.0;
        var rotX = 0.0;
        if( blockDragged != null ) {
           // trace(blockDragged.name);
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
                lastMouseX = lastMouseY = 0;
            }

            if( mouse.wheelDelta != 0 ) {
                if( mouse.wheelDelta < 0 ) {
                    if( cam.transform.loc.y < ZOOM_MIN ) cam.transform.loc.y += ZOOM_SPEED;
                } else {
                    if( cam.transform.loc.y > ZOOM_MAX ) cam.transform.loc.y -= ZOOM_SPEED;
                }
                cam.transform.buildMatrix();
            }
        }
        camRigZ.transform.rotate( new Vec4( 0,0,rotZ,1), 1 );
        var rx = camRigX.transform.rot.x + rotX;
        if( rx > ROT_X_MAX ) rx = ROT_X_MAX;
        else if( rx < ROT_X_MIN ) rx = ROT_X_MIN;
        camRigX.transform.rot.x = rx;
        camRigX.transform.buildMatrix();
    }

    public function resetCamera( delay = 0.0 ) {

        //trace("----"+camRigZ.transform.rot.z);

        //TODO rotation tweens
        
        var t = {
            zoom: cam.transform.loc.y,
            //rx: 0.0, //camRigX.transform.rot.x,
            //rz: camRigZ.transform.rot.z
        };

        Tween.to({
            target: t,
            props: {
                zoom: ZOOM_DEFAULT,
                //rx: Math.abs( camRigX.transform.rot.x ),
                //rz: Math.abs( camRigX.transform.rot.z ) - Math.PI/4
               //rx: 0.0,
                //rz: camRigZ.transform.rot.z - Math.PI/4
                //rz: 0.0 //Math.PI/4
            },
            duration: Math.abs( cam.transform.loc.y - ZOOM_DEFAULT ) * 0.1,
            delay: delay,
            ease: QuadInOut,
            tick: () -> {
                cam.transform.loc.y = t.zoom;
                //cam.transform.buildMatrix();
                //trace(t.rz);
                //camRigZ.transform.rot.set( 0, 0, t.rz, 1 );
                //camRigZ.transform.buildMatrix();
               //camRigZ.transform.rotate( Vec4.zAxis(), rz );
                //camRigX.transform.rot.fromAxisAngle( Vec4.xAxis(), t.rx);
                // camRigZ.transform.rot.fromAxisAngle( Vec4.zAxis(), t.rz );
                // camRigZ.transform.buildMatrix();
                //camRigX.transform.rot.x = t.rx;
                //camRigZ.transform.rot.z = t.rz;
                //camRigX.transform.rotate( Vec4.zAxis(), t.rz );
                //camRigZ.transform.rotate( Vec4.zAxis(), Math.PI/4 );
                ///camRigX.transform.rotate( Vec4.xAxis(), camRigX.transform.rot.x - t.rx );
                ///camRigZ.transform.rotate( Vec4.zAxis(), camRigZ.transform.rot.z - t.rz );
                //camRigZ.transform.rotate( Vec4.zAxis(), camRigZ.transform.rot.z - t.rz );
            }
        });

        camRigX.transform.rot.set(0,0,0,1);
        camRigX.transform.buildMatrix();
        
        //cam.transform.loc.y = ZOOM_DEFAULT;
        //cam.transform.buildMatrix();

        camRigZ.transform.rot.set(0,0,0,1);
        //camRigZ.transform.rot.z = Math.PI/4;
        camRigZ.transform.rotate( Vec4.zAxis(), Math.PI/4 );
        //camRigZ.transform.buildMatrix();
    }

}
