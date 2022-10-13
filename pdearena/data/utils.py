from typing import Tuple, Optional

import torch


def create_data(
    pde,
    scalar_fields: torch.Tensor,
    vector_fields: torch.Tensor,
    grid: Optional[torch.Tensor],
    start: int,
    time_history: int,
    time_future: int,
    time_gap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create training, test, valid data for one step prediction out of trajectory.

    Args:
        scalar_fields (torch.Tensor): input data of the shape [t * pde.n_scalar_components, x, y]
        vector_fields (torch.Tensor): input data of the shape [2 * t * pde.n_vector_components, x, y]
        start_time (list): list of starting points of one batch within the different timepoints of one trajectory

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: input trajectories, label trajectories
    """
    assert pde.n_scalar_components > 0 or pde.n_vector_components > 0

    # Different starting points of one batch according to field_x(t_0), field_y(t_0), ...
    end_time = start + time_history
    target_start_time = end_time + time_gap
    target_end_time = target_start_time + time_future
    if pde.n_scalar_components > 0:
        data_scalar = scalar_fields[start:end_time]
        labels_scalar = scalar_fields[target_start_time:target_end_time]

    if pde.n_vector_components > 0:
        data_vector = vector_fields[start:end_time]
        labels_vector = vector_fields[target_start_time:target_end_time]
        data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)
        labels = torch.cat((labels_scalar, labels_vector), dim=1).unsqueeze(0)
    else:
        data = data_scalar.unsqueeze(0)
        labels = labels_scalar.unsqueeze(0)

    # if grid is not None:
    #     raise NotImplementedError()
    #     data = torch.cat((data, grid), dim=1)

    if labels.size(1) == 0:
        import pdb

        pdb.set_trace()
    return data, labels


def create_time_conditioned_data(
    pde, scalar_fields, vector_fields, grid, start_time: int, end_time: int, delta_t
):
    assert pde.n_scalar_components > 0 or pde.n_vector_components > 0
    if pde.n_scalar_components > 0:
        data_scalar = scalar_fields[start_time : start_time + 1]
        label_scalar = scalar_fields[end_time : end_time + 1]

    if pde.n_vector_components > 0:
        data_vector = vector_fields[start_time : start_time + 1]
        label_vector = vector_fields[end_time : end_time + 1]
        data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)
        labels = torch.cat((label_scalar, label_vector), dim=1).unsqueeze(0)
    if grid is not None:
        data = torch.cat((data, grid), dim=1)

    return data, labels, delta_t


