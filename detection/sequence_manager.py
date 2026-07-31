"""
=========================================================
AnomaliNet Sequence Manager
=========================================================
Maintains one feature sequence per tracked person.
=========================================================
"""

from collections import deque

from detection.config import SEQUENCE_LENGTH


class SequenceManager:

    def __init__(self):

        self.buffers = {}

    # ---------------------------------------------------
    # Add feature for a tracked person
    # ---------------------------------------------------

    def add_feature(self, person_id, feature):

        if person_id not in self.buffers:

            self.buffers[person_id] = deque(
                maxlen=SEQUENCE_LENGTH
            )

        self.buffers[person_id].append(feature)

    # ---------------------------------------------------
    # Check if sequence is ready
    # ---------------------------------------------------

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

        return list(self.buffers[person_id])

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

    # ---------------------------------------------------
    # Clear all buffers
    # ---------------------------------------------------

    def clear(self):

        self.buffers.clear()