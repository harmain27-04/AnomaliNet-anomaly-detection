ANOMALY_THRESHOLD = 0.08

def is_anomaly(score):

    return score > ANOMALY_THRESHOLD