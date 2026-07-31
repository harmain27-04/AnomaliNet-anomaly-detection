import numpy as np

class FeatureFusion:

    def __init__(self):
        pass

    def fuse(
        self,
        spatial_feature,
        temporal_feature
    ):

        spatial_feature = np.array(
            spatial_feature
        )

        temporal_feature = np.array(
            temporal_feature
        )

        fused_feature = np.concatenate(
            [
                spatial_feature,
                temporal_feature
            ],
            axis=0
        )

        return fused_feature