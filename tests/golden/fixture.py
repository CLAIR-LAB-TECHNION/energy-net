"""Load the recorded legacy trajectories (see record_legacy_pcs.py)."""

import gzip
import json
from functools import lru_cache

import numpy as np

from tests.golden.scenarios import FIXTURE_JSON, FIXTURE_NPZ


@lru_cache(maxsize=1)
def load_fixture():
    with gzip.open(FIXTURE_JSON, "rb") as handle:
        metadata = json.loads(handle.read().decode("utf-8"))
    with np.load(FIXTURE_NPZ) as data:
        arrays = {key: data[key] for key in data.files}
    return metadata, arrays
