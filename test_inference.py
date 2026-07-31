import numpy as np

from detection.inference import run_inference

dummy = np.random.rand(
    16,
    2048
)

result = run_inference(
    dummy
)

print(result)