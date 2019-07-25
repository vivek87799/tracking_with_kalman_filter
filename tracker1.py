# Import python libraries
import numpy as np
from kalman_filter.kalman_filter1 import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.prediction2d = ()  # predicted centroids (x,y)
        self.prediction3d = ()
        self.skipped_frames = 0  # number of frames skipped undetected
        self.min_hits = 0
        self.trace = []  # trace path
        self.trace3d = []  # trace 3d path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked

    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_length: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.tracks3d = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except Exception as e:
                    print("[ERROR] error in tracker1.py while calculation the dist", e)

        # Let's average the squared ERROR
        cost = 0.5 * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                # TODO_ with the prediction update the 3d points
                self.tracks[i].KF.correct(detections[assignment[i]], 1)
                self.tracks[i].prediction = self.tracks[i].KF.kf.x
            else:
                # TODO_ with the prediction update the 3d points
                self.tracks[i].KF.correct(self.tracks[i].KF.kf.x, 0)
                self.tracks[i].prediction = self.tracks[i].KF.kf.x


            if len(self.tracks[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

    def Update3d(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if len(self.tracks3d) == 0:
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks3d.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks3d)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks3d)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks3d[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0] +
                                       diff[2][0] * diff[2][0])
                    cost[i][j] = distance
                except Exception as e:
                    print("[ERROR] error in tracker1.py while calculation the dist", e)

        # Let's average the squared ERROR
        cost = 0.5 * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                self.tracks3d[i].min_hits += 1
            else:
                self.tracks3d[i].skipped_frames += 1
                self.tracks3d[i].min_hits = 0

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks3d)):
            if self.tracks3d[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks3d):
                    del self.tracks3d[id]
                    del assignment[id]

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks3d.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks3d[i].KF.predict3d()

            if assignment[i] != -1:
                self.tracks3d[i].skipped_frames = 0
                # get the velocity of the moving object or the diff in pos

                diff_x = detections[assignment[i]][0][0] - self.tracks3d[i].KF.b3d[0][0]
                diff_y = detections[assignment[i]][1][0] - self.tracks3d[i].KF.b3d[1][0]
                diff_z = detections[assignment[i]][2][0] - self.tracks3d[i].KF.b3d[2][0]
                self.tracks3d[i].KF.velocity3d(diff_x, diff_y, diff_z)
                self.tracks3d[i].prediction = self.tracks3d[i].KF.correct3d(
                                            detections[assignment[i]], 1)
            else:
                self.tracks3d[i].prediction = self.tracks3d[i].KF.correct3d(
                                            np.array([[0], [0]]), 0)

            if len(self.tracks3d[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks3d[i].trace) -
                               self.max_trace_length):
                    del self.tracks3d[i].trace[j]

            self.tracks3d[i].trace.append(self.tracks3d[i].prediction)
            self.tracks3d[i].KF.lastResult3d = self.tracks3d[i].prediction
