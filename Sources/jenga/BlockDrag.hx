package jenga;

import iron.Trait;
import iron.system.Input;
import iron.math.Vec3;
import iron.math.Vec4;
import iron.math.Mat4;
import iron.math.RayCaster;
import armory.trait.physics.RigidBody;
import armory.trait.physics.PhysicsWorld;

class BlockDrag extends Trait {
	
	static var start = new Vec4();
	static var end = new Vec4();

	static var v = new Vec4();
	static var m = Mat4.identity();
	static var first = true;

	public static dynamic function onDragStart( obj : Object ) {}
	public static dynamic function onDragEnd( obj : Object ) {}

	@prop public var linearLowerLimit = new Vec3(0,0,0);
	@prop public var linearUpperLimit = new Vec3(0,0,0);
	@prop public var angularLowerLimit = new Vec3(-10,-10,-10);
	@prop public var angularUpperLimit = new Vec3(10,10,10);

	public var enabled = true;
	public var pickedBody(default,null) : RigidBody;
	public var body(default,null) : RigidBody;

	var pickConstraint: bullet.Bt.Generic6DofConstraint;
	var pickDist: Float;

	var rayFrom: bullet.Bt.Vector3;
	var rayTo: bullet.Bt.Vector3;

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
			var drag = b.object.getTrait(BlockDrag);
			//if (b != null && b.mass > 0 && !b.body.isKinematicObject() && b.object.getTrait(PhysicsDrag) != null) {
			if (b != null && b.mass > 0 && !b.body.isKinematicObject() && drag != null && drag.enabled) {

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
				// pickConstraint.setLinearLowerLimit(new bullet.Bt.Vector3(linearLowerLimit.x, linearLowerLimit.y, linearLowerLimit.z));
				// pickConstraint.setLinearUpperLimit(new bullet.Bt.Vector3(linearUpperLimit.x, linearUpperLimit.y, linearUpperLimit.z));
				// pickConstraint.setAngularLowerLimit(new bullet.Bt.Vector3(angularLowerLimit.x, angularLowerLimit.y, angularLowerLimit.z));
				// pickConstraint.setAngularUpperLimit(new bullet.Bt.Vector3(angularUpperLimit.x, angularUpperLimit.y, angularUpperLimit.z));
				pickConstraint.setLinearLowerLimit(new bullet.Bt.Vector3(-0, -0, -0));
				pickConstraint.setLinearUpperLimit(new bullet.Bt.Vector3(0, 0, 0));
				pickConstraint.setAngularLowerLimit(new bullet.Bt.Vector3(-5, -5, -5));
				pickConstraint.setAngularUpperLimit(new bullet.Bt.Vector3(5, 5, 5));
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

				onDragStart( pickedBody.object );
			}
		}

		else if (mouse.released()) {
			if (pickConstraint != null) {
				var obj = pickedBody.object;
				physics.world.removeConstraint(pickConstraint);
				pickConstraint = null;
				pickedBody = null;
				onDragEnd( obj);
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
