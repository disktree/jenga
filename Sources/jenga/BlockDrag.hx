package jenga;

class BlockDrag extends Trait {

    static var start = new Vec4();
	static var end = new Vec4();

    static var v = new Vec4();
	static var m = Mat4.identity();
	static var first = true;
	
    public var pickedBody(default,null) : RigidBody = null;

	var pickConstraint : bullet.Bt.Generic6DofConstraint = null;
	var pickDist : Float;

	var rayFrom : bullet.Bt.Vector3;
	var rayTo : bullet.Bt.Vector3;

	public function new() {
		super();
		if (first) {
			first = false;
			notifyOnUpdate(update);
		}
	}

	function update() {
		var physics = PhysicsWorld.active;
		if (pickedBody != null) pickedBody.activate();

		var mouse = Input.getMouse();
		if (mouse.started()) {

			var b = physics.pickClosest(mouse.x, mouse.y);
			if (b != null && b.mass > 0 && !b.body.isKinematicObject() && b.object.getTrait(BlockDrag) != null) {

				setRays();
				pickedBody = b;

				m.getInverse(b.object.transform.world);
				var hit = physics.hitPointWorld;
				v.setFrom(hit);
				v.applymat4(m);
				var localPivot = new bullet.Bt.Vector3(v.x, v.y, v.z);
				var tr = new bullet.Bt.Transform();
				tr.setIdentity();
				tr.setOrigin(localPivot);

				pickConstraint = new bullet.Bt.Generic6DofConstraint(b.body, tr, false);
				pickConstraint.setLinearLowerLimit(new bullet.Bt.Vector3(0, 0, 0));
				pickConstraint.setLinearUpperLimit(new bullet.Bt.Vector3(0, 0, 0));
				pickConstraint.setAngularLowerLimit(new bullet.Bt.Vector3(-10, -10, -10));
				pickConstraint.setAngularUpperLimit(new bullet.Bt.Vector3(10, 10, 10));
				physics.world.addConstraint(pickConstraint, false);

				/*pickConstraint.setParam(4, 0.8, 0);
				pickConstraint.setParam(4, 0.8, 1);
				pickConstraint.setParam(4, 0.8, 2);
				pickConstraint.setParam(4, 0.8, 3);
				pickConstraint.setParam(4, 0.8, 4);
				pickConstraint.setParam(4, 0.8, 5);

				pickConstraint.setParam(1, 0.1, 0);
				pickConstraint.setParam(1, 0.1, 1);
				pickConstraint.setParam(1, 0.1, 2);
				pickConstraint.setParam(1, 0.1, 3);
				pickConstraint.setParam(1, 0.1, 4);
				pickConstraint.setParam(1, 0.1, 5);*/

				pickDist = v.set(hit.x - rayFrom.x(), hit.y - rayFrom.y(), hit.z - rayFrom.z()).length();

				Input.occupied = true;
			}
		}

		else if (mouse.released()) {
			if (pickConstraint != null) {
				physics.world.removeConstraint(pickConstraint);
				pickConstraint = null;
				pickedBody = null;
			}
			Input.occupied = false;
		}

		else if (mouse.down()) {
			if (pickConstraint != null) {
				setRays();

				// Keep it at the same picking distance
				var dir = new bullet.Bt.Vector3(rayTo.x() - rayFrom.x(), rayTo.y() - rayFrom.y(), rayTo.z() - rayFrom.z());
				dir.normalize();
				dir.setX(dir.x() * pickDist);
				dir.setY(dir.y() * pickDist);
				dir.setZ(dir.z() * pickDist);
				var newPivotB = new bullet.Bt.Vector3(rayFrom.x() + dir.x(), rayFrom.y() + dir.y(), rayFrom.z() + dir.z());

				#if (js || hl)
				pickConstraint.getFrameOffsetA().setOrigin(newPivotB);
				#elseif cpp
				pickConstraint.setFrameOffsetAOrigin(newPivotB);
				#end
			}
		}
	}

	inline function setRays() {
		var mouse = Input.getMouse();
		var camera = iron.Scene.active.camera;
		var v = camera.transform.world.getLoc();
		rayFrom = new bullet.Bt.Vector3(v.x, v.y, v.z);
		RayCaster.getDirection(start, end, mouse.x, mouse.y, camera);
		rayTo = new bullet.Bt.Vector3(end.x, end.y, end.z);
	}
}
