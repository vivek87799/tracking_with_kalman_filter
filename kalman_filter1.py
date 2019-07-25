# Import python libraries
import logging
import inspect
import numpy as np
from filterpy.kalman import KalmanFilter as KalmanFilterpy
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

# logging.basicConfig(filename='posvel.txt', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 0.2   # time step

    tracker.F = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                          [0.0, 0.0, 0.0, 1, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 1]])
    # tracker.u = 0.
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001)
    tracker.Q = block_diag(q, q)

    tracker.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0]])
    tracker.R = np.diag((0.1, 0.1, 0.1))
    tracker.x = np.array([[0, 0, 0, 0, 0, 0]]).T
    tracker.P = np.eye(6) * 25
    return tracker


tracker = KalmanFilterpy(dim_x=6, dim_z=3)
dt = 0.2

tracker.F = np.array([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                           [0.0, 0.0, 0.0, 1, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1]])


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class
        Return:
            None
        """

        self.kf = KalmanFilterpy(dim_x=6, dim_z=3)
        self.dt = 0.2  # time step
        """
        self.kf.F = np.array([[1.0, 0.0, 0.0, self.dt, 0.0,     0.0],
                              [0.0, 1.0, 0.0, 0.0,     self.dt, 0.0],
                              [0.0, 0.0, 1.0, 0.0,     0.0,     self.dt],
                              [0.0, 0.0, 0.0, 1,       0.0,     0.0],
                              [0.0, 0.0, 0.0, 0.0,     1,       0.0],
                              [0.0, 0.0, 0.0, 0.0,     0.0,     1]])
        """
        self.kf.F = np.array([[1.0, dt, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, dt, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 1, dt],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 1]])
        # tracker.u = 0.
        q = Q_discrete_white_noise(dim=3, dt=dt, var=1)
        self.kf.Q = block_diag(q, q)  # process noise matrix

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]])
        self.kf.R = np.diag((0.01, 0.01, 0.01))   # observation noise matrix
        self.kf.x = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.kf.P = np.eye(6) * 4
        self.last_result = 0.0

    def predict(self):
        """

        :return:
        """
        global _module_name
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            self.last_result = self.kf.predict()
            return self.kf.predict()
        except Exception as error:
            logger.error(f"Error in {_module_name} module, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    def correct(self, b, flag):
        """

        :param b:
        :param flag:
        :return:
        """
        global _module_name
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            if not flag:
                return self.kf.update(self.lastResult)
            else:
                return self.kf.update(b)
        except Exception as error:
            logger.info(f"Error in {_module_name}, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")




