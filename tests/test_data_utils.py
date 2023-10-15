import pytest
import torch

from pdearena.data.utils import create_data2D, create_maxwell_data


@pytest.mark.parametrize(
    "n_input_scalar_components,n_input_vector_components,n_output_scalar_components,n_output_vector_components",
    [(1, 1, 1, 1), (2, 0, 2, 0), (0, 1, 0, 1), (2, 1, 1, 1)],
)
@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("time_history", [1, 2, 3, 4])
@pytest.mark.parametrize("time_future", [1, 2, 3, 4])
def test_create_data2D(
    n_input_scalar_components,
    n_input_vector_components,
    n_output_scalar_components,
    n_output_vector_components,
    start,
    time_history,
    time_future,
):
    T = 15
    N = 64
    C = n_input_scalar_components + n_input_vector_components * 2
    n_scalar = max(n_input_scalar_components, n_output_scalar_components)
    n_vector = max(n_input_vector_components * 2, n_output_vector_components * 2)
    scalar_fields = torch.rand(T, n_scalar, N, N)
    vector_fields = torch.rand(
        T,
        n_vector,
        N,
        N,
    )
    grid = None
    time_gap = 0
    data, targets = create_data2D(
        n_input_scalar_components,
        n_input_vector_components,
        n_output_scalar_components,
        n_output_vector_components,
        scalar_fields,
        vector_fields,
        grid,
        start,
        time_history,
        time_future,
        time_gap,
    )
    assert data.shape == (1, time_history, C, N, N)
    assert targets.shape == (1, time_future, n_output_scalar_components + n_output_vector_components * 2, N, N)
    torch.testing.assert_close(
        scalar_fields[start : start + time_history, :n_input_scalar_components, ...],
        data[0, :, :n_input_scalar_components, ...],
    )
    torch.testing.assert_close(
        vector_fields[start : start + time_history, : n_input_vector_components * 2, ...],
        data[0, :, n_input_scalar_components : n_input_scalar_components + 2 * n_input_vector_components, ...],
    )
    torch.testing.assert_close(
        scalar_fields[
            start + time_history + time_gap : start + time_history + time_gap + time_future,
            :n_output_scalar_components,
            ...,
        ],
        targets[0, :, :n_output_scalar_components, ...],
    )
    torch.testing.assert_close(
        vector_fields[
            start + time_history + time_gap : start + time_history + time_gap + time_future,
            : n_output_vector_components * 2,
            ...,
        ],
        targets[0, :, n_output_scalar_components : n_output_scalar_components + 2 * n_output_vector_components, ...],
    )


@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("time_history", [1, 2, 3, 4])
@pytest.mark.parametrize("time_future", [1, 2, 3, 4])
def test_create_maxwell_data(
    start,
    time_history,
    time_future,
):
    T = 15
    N = 64
    d_field = torch.rand(T, 3, N, N, N)
    h_field = torch.rand(T, 3, N, N, N)
    time_gap = 0
    data, targets = create_maxwell_data(
        d_field,
        h_field,
        start,
        time_history,
        time_future,
        time_gap,
    )
    assert data.shape == (1, time_history, 6, N, N, N)
    assert targets.shape == (1, time_future, 6, N, N, N)

    torch.testing.assert_close(d_field[start : start + time_history], data[0, :, :3])
    torch.testing.assert_close(h_field[start : start + time_history], data[0, :, 3:])
    torch.testing.assert_close(
        d_field[start + time_history + time_gap : start + time_history + time_gap + time_future, :3],
        targets[0, :, :3],
    )
    torch.testing.assert_close(
        h_field[start + time_history + time_gap : start + time_history + time_gap + time_future, :3],
        targets[0, :, 3:],
    )
