from .oned.datapipes.kuramotosivashinsky1d import (
    onestep_test_datapipe_ks,
    onestep_valid_datapipe_ks,
    train_datapipe_ks,
    trajectory_test_datapipe_ks,
    trajectory_valid_datapipe_ks,
)
from .threed.datapipes import (
    onestep_test_datapipe_maxwell,
    onestep_valid_datapipe_maxwell,
    rollout_test_datapipe_maxwell,
    rollout_valid_datapipe_maxwell,
    train_datapipe_maxwell,
)
from .twod.datapipes.navierstokes2d import (
    onestep_test_datapipe_ns,
    onestep_test_datapipe_ns_cond,
    onestep_valid_datapipe_ns,
    onestep_valid_datapipe_ns_cond,
    train_datapipe_ns,
    train_datapipe_ns_cond,
    trajectory_test_datapipe_ns,
    trajectory_test_datapipe_ns_cond,
    trajectory_valid_datapipe_ns,
)
from .twod.datapipes.shallowwater2d import (
    onestep_test_datapipe_1day_vel,
    onestep_test_datapipe_1day_vort,
    onestep_test_datapipe_2day_vel,
    onestep_test_datapipe_2day_vort,
    onestep_valid_datapipe_1day_vel,
    onestep_valid_datapipe_1day_vort,
    onestep_valid_datapipe_2day_vel,
    onestep_valid_datapipe_2day_vort,
    train_datapipe_1day_vel,
    train_datapipe_1day_vort,
    train_datapipe_2day_vel,
    train_datapipe_2day_vort,
    trajectory_test_datapipe_1day_vel,
    trajectory_test_datapipe_1day_vort,
    trajectory_test_datapipe_2day_vel,
    trajectory_test_datapipe_2day_vort,
    trajectory_valid_datapipe_1day_vel,
    trajectory_valid_datapipe_1day_vort,
    trajectory_valid_datapipe_2day_vel,
    trajectory_valid_datapipe_2day_vort,
)

DATAPIPE_REGISTRY = {}

DATAPIPE_REGISTRY["KuramotoSivashinsky1D"] = {}
DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]["train"] = train_datapipe_ks
DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]["valid"] = [onestep_valid_datapipe_ks, trajectory_valid_datapipe_ks]
DATAPIPE_REGISTRY["KuramotoSivashinsky1D"]["test"] = [onestep_test_datapipe_ks, trajectory_test_datapipe_ks]

DATAPIPE_REGISTRY["NavierStokes2D"] = {}
DATAPIPE_REGISTRY["NavierStokes2D"]["train"] = train_datapipe_ns
DATAPIPE_REGISTRY["NavierStokes2D"]["valid"] = [onestep_valid_datapipe_ns, trajectory_valid_datapipe_ns]
DATAPIPE_REGISTRY["NavierStokes2D"]["test"] = [onestep_test_datapipe_ns, trajectory_test_datapipe_ns]

DATAPIPE_REGISTRY["Cond-NavierStokes2D"] = {}
DATAPIPE_REGISTRY["Cond-NavierStokes2D"]["train"] = train_datapipe_ns_cond
DATAPIPE_REGISTRY["Cond-NavierStokes2D"]["valid"] = onestep_valid_datapipe_ns_cond
DATAPIPE_REGISTRY["Cond-NavierStokes2D"]["test"] = [onestep_test_datapipe_ns_cond, trajectory_test_datapipe_ns_cond]

DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"]["train"] = train_datapipe_2day_vel
DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"]["valid"] = [
    onestep_valid_datapipe_2day_vel,
    trajectory_valid_datapipe_2day_vel,
]
DATAPIPE_REGISTRY["ShallowWater2DVel-2Day"]["test"] = [
    onestep_test_datapipe_2day_vel,
    trajectory_test_datapipe_2day_vel,
]

DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"]["train"] = train_datapipe_2day_vort
DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"]["valid"] = [
    onestep_valid_datapipe_2day_vort,
    trajectory_valid_datapipe_2day_vort,
]
DATAPIPE_REGISTRY["ShallowWater2DVort-2Day"]["test"] = [
    onestep_test_datapipe_2day_vort,
    trajectory_test_datapipe_2day_vort,
]

DATAPIPE_REGISTRY["ShallowWater2DVel-1Day"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVel-1Day"]["train"] = train_datapipe_1day_vel
DATAPIPE_REGISTRY["ShallowWater2DVel-1Day"]["valid"] = [
    onestep_valid_datapipe_1day_vel,
    trajectory_valid_datapipe_1day_vel,
]
DATAPIPE_REGISTRY["ShallowWater2DVel-1Day"]["test"] = [
    onestep_test_datapipe_1day_vel,
    trajectory_test_datapipe_1day_vel,
]

DATAPIPE_REGISTRY["ShallowWater2DVort-1Day"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVort-1Day"]["train"] = train_datapipe_1day_vort
DATAPIPE_REGISTRY["ShallowWater2DVort-1Day"]["valid"] = [
    onestep_valid_datapipe_1day_vort,
    trajectory_valid_datapipe_1day_vort,
]
DATAPIPE_REGISTRY["ShallowWater2DVort-1Day"]["test"] = [
    onestep_test_datapipe_1day_vort,
    trajectory_test_datapipe_1day_vort,
]

DATAPIPE_REGISTRY["Maxwell3D"] = {}
DATAPIPE_REGISTRY["Maxwell3D"]["train"] = train_datapipe_maxwell
DATAPIPE_REGISTRY["Maxwell3D"]["valid"] = [onestep_valid_datapipe_maxwell, rollout_valid_datapipe_maxwell]
DATAPIPE_REGISTRY["Maxwell3D"]["test"] = [onestep_test_datapipe_maxwell, rollout_test_datapipe_maxwell]
