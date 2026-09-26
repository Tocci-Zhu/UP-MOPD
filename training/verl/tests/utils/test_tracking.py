# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
import pytest
import torch

from verl.utils.tracking import FileLogger


def test_file_logger_serializes_scalar_torch_and_numpy_metrics(tmp_path, monkeypatch) -> None:
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(path))
    logger = FileLogger("project", "experiment")

    logger.log(
        {
            "torch_float": torch.tensor(1.5),
            "torch_int": torch.tensor(2),
            "torch_bool": torch.tensor(True),
            "numpy_float": np.float32(3.5),
            "nested": {"numpy_int": np.int64(4), "values": [np.bool_(True)]},
        },
        step=np.int64(7),
    )
    logger.finish()

    assert json.loads(path.read_text()) == {
        "step": 7,
        "data": {
            "torch_float": 1.5,
            "torch_int": 2,
            "torch_bool": True,
            "numpy_float": 3.5,
            "nested": {"numpy_int": 4, "values": [True]},
        },
    }


@pytest.mark.parametrize("shape", [(1,), (2, 2)])
def test_file_logger_rejects_non_scalar_tensors_without_partial_line(tmp_path, monkeypatch, shape) -> None:
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(path))
    logger = FileLogger("project", "experiment")

    with pytest.raises(TypeError, match=r"only supports scalar tensors; got shape="):
        logger.log({"bad": torch.ones(shape)}, step=1)
    logger.finish()

    assert path.read_text() == ""
