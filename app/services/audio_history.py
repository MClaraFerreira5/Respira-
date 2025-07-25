from datetime import datetime
import numpy as np


class AudioHistory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.predictions = []
        self.audio_segments = []
        self.timestamps = []

    def add_prediction(self, prediction, audio_segment=None):
        self.predictions.append(prediction)
        self.audio_segments.append(audio_segment)
        self.timestamps.append(datetime.now())

        if len(self.predictions) > self.max_length:
            self.predictions.pop(0)
            self.audio_segments.pop(0)
            self.timestamps.pop(0)

    def smooth(self):
        if not self.predictions:
            return np.zeros(521)
        return np.mean(self.predictions, axis=0)
