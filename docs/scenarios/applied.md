# Scenario: You have a new class of PDE and want to see if PDE surrogates can be useful for you

There are hundreds of possible PDE surrogate models. As with most deep learning, choice of hyperparameters can matter a lot too.
You might also have your own runtime requirements, and data or time limitations.
PDEArena provides a _simple interface_ for you to try out many possible model designs and understand the best model for your task and constraints.
All you need to do is, define your PDE configuration details in [`pde.py`](https://github.com/microsoft/pdearena/blob/main/pdedatagen/pde.py), write a `IterDataPipe` for your dataset, add it to the [`DATAPIPE_REGISTRY`](https://github.com/microsoft/pdearena/blob/main/pdearena/data/registry.py), and you should be good to go.

## Simple Example

Let's say you want a new task modeling Shallow water making predictions at 18 hours interval.
You can subclass the [`ShallowWaterDatasetOpener`][pdearena.data.twod.datapipes.shallowwater2d.ShallowWaterDatasetOpener] as:

```py
class ShallowWaterDatasetOpener18Hr(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        # sample_rate=1 implies 6hr
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=3)
```

Now you can set up the various datapipes for training, validation and testing:

```py
# Train
train_datapipe_18Hr_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener18Hr,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
# Valid
onestep_valid_datapipe_18Hr_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener18Hr,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_18Hr_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener18Hr,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
# Test
onestep_test_datapipe_18Hr_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener18Hr,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_18Hr_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener18Hr,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)
```

Then you can add these datapipes to the data [`registry`](https://github.com/microsoft/pdearena/blob/main/pdearena/data/registry.py):
```py
DATAPIPE_REGISTRY["ShallowWater2DVel-18Hr"] = {}
DATAPIPE_REGISTRY["ShallowWater2DVel-18Hr"]["train"] = train_datapipe_18Hr_vel
DATAPIPE_REGISTRY["ShallowWater2DVel-18Hr"]["valid"] = [
    onestep_valid_datapipe_18Hr_vel,
    trajectory_valid_datapipe_18Hr_vel,
]
DATAPIPE_REGISTRY["ShallowWater2DVel-18Hr"]["test"] = [
    onestep_test_datapipe_18Hr_vel,
    trajectory_test_datapipe_18Hr_vel,
]
```

Finally you can train different models from the model zoo by setting the `data.task=ShallowWater2DVel-18Hr`:
```yaml
data:
  task: ShallowWater2DVel-18Hr
```
See [`config`](https://github.com/microsoft/pdearena/blob/main/configs/shallowwater2d_2day.yaml) for an example of training with 2-day prediction.
