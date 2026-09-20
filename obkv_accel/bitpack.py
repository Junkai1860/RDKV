"""
Minimal uint8 packing helpers used by ``obkv_accel.packing``.

These are copied from the original RDKV compress utilities so the
benchmark bundle does not need to import the heavier ``rdkv`` package.
"""

from __future__ import annotations

import torch


def transfer_8bit_to_2bit_batchwise(input: torch.Tensor) -> torch.Tensor:
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 4 == 0
    size = input.shape[-1] // 4
    input[..., 0:size] = (
        input[..., 0:size]
        + input[..., size : 2 * size] * (2 ** 2)
        + input[..., 2 * size : 3 * size] * (2 ** 4)
        + input[..., 3 * size :] * (2 ** 6)
    )
    return input[..., 0:size].clone()


def transfer_2bit_to_8bit_batchwise(input: torch.Tensor) -> torch.Tensor:
    assert input.dtype == torch.uint8
    low_end = input & 3
    mid_low_end = (input >> 2) & 3
    mid_high_end = (input >> 4) & 3
    high_end = (input >> 6) & 3
    return torch.cat((low_end, mid_low_end, mid_high_end, high_end), dim=-1)


def transfer_8bit_to_4bit_batchwise(input: torch.Tensor) -> torch.Tensor:
    assert input.dtype == torch.uint8
    assert input.shape[-1] % 2 == 0
    size = input.shape[-1] // 2
    input[..., 0:size] = input[..., 0:size] + input[..., size:] * (2 ** 4)
    return input[..., 0:size].clone()


def transfer_4bit_to_8bit_batchwise(input: torch.Tensor) -> torch.Tensor:
    assert input.dtype == torch.uint8
    low_end = input % (2 ** 4)
    high_end = (input - low_end) / (2 ** 4)
    return torch.cat((low_end, high_end), dim=-1)
