from models.fusion.feature_fusion import FeatureFusion
import numpy as np

fusion = FeatureFusion()

spatial = np.random.rand(2048)
temporal = np.random.rand(512)

result = fusion.fuse(
    spatial,
    temporal
)

print(result.shape)