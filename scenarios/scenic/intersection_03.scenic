"""
TITLE: Intersection 03
AUTHOR: Francis Indaheng, findaheng@berkeley.edu
DESCRIPTION: Ego vehicle either goes straight or makes a left turn at
4-way intersection and must suddenly stop to avoid collision when
adversary vehicle from lateral lane continues straight.
SOURCE: NHSTA, #28 #29
"""

#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../maps/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model

#################################
# CONSTANTS                     #
#################################

MODEL = 'vehicle.tesla.model3'

EGO_INIT_DIST = [20, 25]
param EGO_SPEED = VerifaiRange(7, 10)
param EGO_BRAKE = VerifaiRange(0.5, 1.0)

ADV_INIT_DIST = [15, 20]
param ADV_SPEED = VerifaiRange(7, 10)

param SAFETY_DIST = VerifaiRange(10, 20)
CRASH_DIST = 5
TERM_DIST = 70

#################################
# AGENT BEHAVIORS               #
#################################

behavior EgoBehavior(trajectory):
	try:
		do FollowTrajectoryBehavior(target_speed=globalParameters.EGO_SPEED, trajectory=trajectory)
	interrupt when withinDistanceToAnyObjs(self, globalParameters.SAFETY_DIST):
		take SetBrakeAction(globalParameters.EGO_BRAKE)
	interrupt when withinDistanceToAnyObjs(self, CRASH_DIST):
		terminate

#################################
# SPATIAL RELATIONS             #
#################################

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m:
		m.type in (ManeuverType.STRAIGHT, ManeuverType.LEFT_TURN),
		egoInitLane.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoInitLane.centerline

advInitLane = Uniform(*filter(lambda m:
		m.type is ManeuverType.STRAIGHT,
		Uniform(*filter(lambda m:
			m.type is ManeuverType.STRAIGHT,
			egoInitLane.maneuvers)
		).conflictingManeuvers)
	).startLane
advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, advInitLane.maneuvers))
advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
advSpawnPt = OrientedPoint in advInitLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = Car at egoSpawnPt,
    with name "sut",
    with rolename "sut",
	with blueprint MODEL,
	with behavior EgoBehavior(egoTrajectory)

adversary = Car at advSpawnPt,
    with name "adv_1",
    with rolename "adv_1",
	with blueprint MODEL,
	with behavior FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, trajectory=advTrajectory)

require EGO_INIT_DIST[0] <= (distance to intersection) <= EGO_INIT_DIST[1]
require ADV_INIT_DIST[0] <= (distance from adversary to intersection) <= ADV_INIT_DIST[1]
terminate when (distance to egoSpawnPt) > TERM_DIST