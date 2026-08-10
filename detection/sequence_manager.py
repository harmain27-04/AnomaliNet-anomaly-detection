"""
=========================================================
AnomaliNet Sequence Manager
=========================================================
Maintains one feature sequence per tracked person.
=========================================================
"""

from collections import deque
import numpy as np
from detection.config import SEQUENCE_LENGTH


class SequenceManager:

    def __init__(self):

        self.buffers = {}

        self.last_update = {}

        self.max_idle_frames = 45

    # ---------------------------------------------------
    # Add feature for a tracked person
    # ---------------------------------------------------

    def add_feature(
    self,
    person_id,
    feature,
    frame_number
    ):

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        if feature.shape != (2048,):

            return

        if person_id not in self.buffers:

            self.buffers[person_id] = deque(
                maxlen=SEQUENCE_LENGTH
            )

        self.buffers[person_id].append(
            feature
        )

        self.last_update[person_id] = frame_number

    # ---------------------------------------------------
    # Check if sequence is ready
    # ---------------------------------------------------
    def cleanup(self, current_frame):

        remove_ids = []

        for person_id in self.last_update:

            if (
                current_frame
                -
                self.last_update[person_id]
                >
                self.max_idle_frames
            ):

                remove_ids.append(
                    person_id
                )

        for person_id in remove_ids:

            self.remove(person_id)
    def is_ready(self, person_id):

        if person_id not in self.buffers:
            return False

        return len(self.buffers[person_id]) == SEQUENCE_LENGTH

    # ---------------------------------------------------
    # Get sequence
    # ---------------------------------------------------

    def get_sequence(self, person_id):

        if not self.is_ready(person_id):

            return None

        sequence = np.array(

            self.buffers[person_id],

            dtype=np.float32

        )

        return sequence

    # ---------------------------------------------------
    # Current length
    # ---------------------------------------------------

    def length(self, person_id):

        if person_id not in self.buffers:
            return 0

        return len(self.buffers[person_id])

    # ---------------------------------------------------
    # Remove inactive person
    # ---------------------------------------------------

    def remove(self, person_id):

        if person_id in self.buffers:

            del self.buffers[person_id]

        if person_id in self.last_update:

            del self.last_update[person_id]

    # ---------------------------------------------------
    # Clear all buffers
    # ---------------------------------------------------

    def clear(self):

        self.buffers.clear()