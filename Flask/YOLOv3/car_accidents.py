import numpy as np


def check_overlap(first_car, second_car, one_diag, second_diag, k):
    dist = np.sqrt((first_car[0] - second_car[0]) ** 2 + (first_car[1] - second_car[1]) ** 2)
    threshold = one_diag + second_diag
    if (dist < threshold * k):
        check = True
    else:
        check = False
    return check


##---angle anomalies functions#
# check anomaly in trajectory by analyzing angular acceleration
def check_angle_anomaly(angle_list_1st, frame, check_frames):
    # iterating and calculating numerical second derivative for list of angles
    angle_change = angle_list_1st[frame - check_frames: frame + check_frames]
    if len(angle_change) > 0:
        diff = max(angle_change) - min(angle_change)
    else:
        diff = 0

    return diff


# checks crash angle at some frame
def check_crash_angle(angle_1st_car, angle_2nd_car, threshold):
    if (angle_1st_car - angle_2nd_car) > threshold:
        check = True
    else:
        check = False
    return check
