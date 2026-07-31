"""
=========================================================
Inference Manager
Controls how frequently AI inference is performed
for each tracked person.
=========================================================
"""


class InferenceManager:

    def __init__(self, interval=3):

        self.interval = interval

        self.frame_counter = {}

    def should_run(self, person_id):

        if person_id not in self.frame_counter:

            self.frame_counter[person_id] = 0

            return True

        self.frame_counter[person_id] += 1

        if self.frame_counter[person_id] >= self.interval:

            self.frame_counter[person_id] = 0

            return True

        return False

    def remove(self, person_id):

        if person_id in self.frame_counter:

            del self.frame_counter[person_id]

    def clear(self):

        self.frame_counter.clear()