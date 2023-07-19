from typing import Optional
import random
import h5py
import torch
import torchdata.datapipes as dp
from pdearena.data.utils import PDEDataConfig
import pdearena.data.utils as datautils


@torch.utils.data.functional_datapipe("read_emfields")
class PDEDatasetOpener3D(dp.iter.IterDataPipe):
    def __init__(
        self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False
    ) -> None:
        super().__init__()
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        if usegrid:
            raise NotImplementedError("3D grids")

    def __iter__(self):
        for path in self.dp:
            f = h5py.File(path, "r")
            data = f[self.mode]
            if self.limit_trajectories is None or self.limit_trajectories == -1:
                num = data["d_field"].shape[0]
            else:
                num = self.limit_trajectories

            # Different workers should be using different trajectory batches
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                num_workers = min(worker_info.num_workers, num)
                per_worker = int(num / float(num_workers))
                iter_start = worker_info.id * per_worker
                iter_end = min(iter_start + per_worker, num)
            else:
                iter_start = 0
                iter_end = num

            for idx in range(iter_start, iter_end):
                d_field = torch.tensor(data["d_field"][idx])
                h_field = torch.tensor(data["h_field"][idx])
                # to T, C, X, Y, Z
                d_field = d_field.permute(
                    0,
                    4,
                    1,
                    2,
                    3,
                )
                h_field = h_field.permute(
                    0,
                    4,
                    1,
                    2,
                    3,
                )
                # d_field = d_field.reshape(
                #     d_field.shape[0], d_field.shape[-1], *d_field.shape[-4:-1]
                # )
                # h_field = h_field.reshape(
                #     h_field.shape[0], h_field.shape[-1], *h_field.shape[-4:-1]
                # )

                yield d_field.float(), h_field.float(), None


class RandomizedPDETrainData3D(dp.iter.IterDataPipe):
    def __init__(
        self, dp, pde: PDEDataConfig, time_history: int, time_future: int, time_gap: int
    ) -> None:
        super().__init__()
        self.dp = dp
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

    def __iter__(self):
        # Length of trajectory
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap

        for (d, h, _) in self.dp:
            # Choose initial random time point at the PDE solution manifold
            start_time = random.choices(
                [t for t in range(self.pde.skip_nt, max_start_time + 1)], k=1
            )
            yield datautils.create_data3D(
                self.pde, d, h, start_time[0], self.time_history, self.time_future, self.time_gap
            )


class PDEEvalTimeStepData3D(dp.iter.IterDataPipe):
    def __init__(
        self, dp, pde: PDEDataConfig, time_history: int, time_future: int, time_gap: int
    ) -> None:
        super().__init__()
        self.dp = dp
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

    def __iter__(self):
        # Length of trajectory
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap
        # We ignore these timesteps in the testing
        start_time = [
            t for t in range(self.pde.skip_nt, max_start_time + 1, self.time_gap + self.time_future)
        ]
        for start in start_time:
            for (d, h, _) in self.dp:
                end_time = start + self.time_history
                target_start_time = end_time + self.time_gap
                target_end_time = target_start_time + self.time_future
                data_dfield = torch.Tensor()
                labels_dfield = torch.Tensor()
                data_hfield = torch.Tensor()
                labels_hfield = torch.Tensor()
                data_dfield = d[
                    start:end_time,
                    ...,
                ]
                labels_dfield = d[
                    target_start_time:target_end_time,
                    ...,
                ]
                data_hfield = h[
                    start:end_time,
                    ...,
                ]
                labels_hfield = h[
                    target_start_time:target_end_time,
                    ...,
                ]

                data = torch.cat((data_dfield, data_hfield), dim=1).unsqueeze(0)  # add batch dim
                labels = torch.cat((labels_dfield, labels_hfield), dim=1).unsqueeze(
                    0
                )  # add batch dim
                # data = data.reshape(1, -1, *data.shape[3:])
                # labels = labels.reshape(1, -1, *labels.shape[3:])
                yield data, labels