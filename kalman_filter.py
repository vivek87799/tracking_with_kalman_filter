# Import python libraries
import logging
import inspect
import numpy as np
from config import delta_time

# logging.basicConfig(filename='posvel.txt', level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

        self.dt = 0.2  # self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        # For 2d

        self.A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # matrix in observation equations
        self.u = np.zeros((4, 1))  # previous state vector

        self.u1 = np.zeros((2, 1))  # position
        self.v1 = np.zeros((2, 1))  # velocity

        # (x,y) tracking object center
        self.b = np.array([[0], [0], [255], [255]])  # vector of observations

        self.P = np.diag((25.0, 25.0, 4.0, 4.0))  # np.diag((3.0, 3.0, 3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, 0.0, self.dt, 0.0], [0.0, 1.0, 0, self.dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1]])
        # state transition mat

        self.Q = np.diag((100, 100, 100, 100))  # self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.diag((0.1, 0.1, 0.1, 0.1))  # np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

        # For 3d

        self.A3d = np.identity((6))  # matrix in observation equations
        self.u3d = np.zeros((6, 1))  # previous state vector

        self.u13d = np.zeros((3, 1))  # position
        self.v13d = np.zeros((3, 1))  # velocity

        # (x,y) tracking object center
        self.b3d = np.array([[0], [0], [0], [255], [255], [255]])  # vector of observations

        self.P3d = np.diag((25.0, 25.0, 25.0, 4.0, 4.0, 4.0))  # np.diag((3.0, 3.0, 3.0, 3.0))  # covariance matrix
        self.F3d = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                           [0.0, 0.0, 0.0, 1, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1]])
        # state transition mat

        self.Q3d = np.diag((1, 1, 1, 1, 1, 1))  # self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R3d = np.diag((0.02, 0.02, 0.02, 0.02, 0.02, 0.02))  # np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult3d = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Return:
            vector of predicted state estimate
        """
        global _module_name
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            # logging.info("%f, %f, %f, %f ", self.u[0][0], self.u[1][0], self.u[2][0], self.u[3][0])
            # Predicted state estimate
            self.lastResult = self.u  # same last predicted result
            self.u = np.round(np.dot(self.F, self.u))
            # Predicted estimate covariance
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
            # self.lastResult = self.u  # same last predicted result
            return self.u
        except Exception as error:
            logger.error(f"Error in {_module_name} module, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """
        global _module_name
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            if not flag:  # update using prediction
                # If no measurement detected use the last estimated state
                self.b = self.lastResult
                self.R = np.diag((100, 100, 100, 100))  # np.eye(self.b.shape[0])  # observation noise matrix
            else:  # update using detection
                self.b = np.array([b[0], b[1], self.v1[0], self.v1[1]])  # b
                np.put(self.b, [0, 1], [b[0], b[1]])
                self.R = np.diag((1, 1, 1, 1))  # np.eye(self.b.shape[0])  # observation noise matrix

            C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
            K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

            self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                                  self.u))))

            """logging.info(" %d, %f, %f, %f, %f : %f, %f, %f, %f", flag, self.b[0][0], self.b[1][0], self.b[2][0],
                         self.b[3][0],
                         self.u[0][0], self.u[1][0], self.u[2][0], self.u[3][0])
            """
            # Process covariance matrix represents error in estimate or process
            self.P = self.P - np.dot(np.dot(K, self.A.T), self.P)  # self.P = self.P - np.dot(K, np.dot(C, K.T))
            self.lastResult = self.u  # [0:2].reshape(2, 1)
            return self.lastResult[0:2].reshape(2, 1)  # return self.u
            # for 2d
            # self.lastResult = self.u1
            # return self.u1
        except Exception as error:
            logger.info(f"Error in {_module_name}, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    def velocity(self, diff_x, diff_y):
        vel_x = diff_x / self.dt
        vel_y = diff_y / self.dt
        self.v1 = np.array([[vel_x], [vel_y]])
        # np.put(self.u, [2, 3], [vel_x, vel_y])
        np.put(self.b, [2, 3], [vel_x, vel_y])

    def predict3d(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Return:
            vector of predicted state estimate
        """
        global _module_name
        try:
            global delta_time
            logger.info(delta_time)
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            # logging.info("%f, %f, %f, %f ", self.u[0][0], self.u[1][0], self.u[2][0], self.u[3][0])
            # Predicted state estimate
            self.lastResult3d = self.u3d  # same last predicted result
            self.u3d = np.round(np.dot(self.F3d, self.u3d))
            # Predicted estimate covariance
            self.P3d = np.dot(self.F3d, np.dot(self.P3d, self.F3d.T)) + self.Q3d
            # self.lastResult = self.u  # same last predicted result
            return self.u3d
        except Exception as error:
            logger.error(f"Error in {_module_name} module, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    def correct3d(self, b3d, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """
        global _module_name
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            if not flag:  # update using prediction
                # If no measurement detected use the last estimated state
                self.b3d = self.lastResult3d
                self.R3d = np.diag((100, 100, 100, 100, 100, 100))  # np.eye(self.b.shape[0])  # observation noise matrix
            else:  # update using detection
                self.b3d = np.array([b3d[0], b3d[1], b3d[2], self.v13d[0], self.v13d[1], self.v13d[2]])  # b
                np.put(self.b3d, [0, 1, 2], [b3d[0], b3d[1], b3d[2]])
                self.R3d = np.diag((0.02, 0.02, 0.02, 1, 1, 1))  # np.eye(self.b.shape[0])  # observation noise matrix

            C = np.dot(self.A3d, np.dot(self.P3d, self.A3d.T)) + self.R3d
            K = np.dot(self.P3d, np.dot(self.A3d.T, np.linalg.inv(C)))
            C_ = np.dot(self.A3d, np.dot(self.P3d, self.A3d.T))
            K_ = np.dot(self.P3d, np.dot(self.A3d.T, np.linalg.inv(C_)))
            U_ = self.u3d + np.dot(K_, (self.b3d - np.dot(self.A3d, self.u3d)))
            self.u3d = self.u3d + np.dot(K, (self.b3d - np.dot(self.A3d, self.u3d)))
            # self.u3d = np.round(self.u3d + np.dot(K, (self.b3d - np.dot(self.A3d, self.u3d))))

            """logging.info(" %d, %f, %f, %f, %f : %f, %f, %f, %f", flag, self.b[0][0], self.b[1][0], self.b[2][0],
                         self.b[3][0],
                         self.u[0][0], self.u[1][0], self.u[2][0], self.u[3][0])
            """
            # Process covariance matrix represents error in estimate or process
            self.P3d = self.P3d - np.dot(np.dot(K, self.A3d.T), self.P3d)  # self.P = self.P - np.dot(K, np.dot(C, K.T))
            self.lastResult3d = self.u3d  # [0:2].reshape(2, 1)
            return self.lastResult3d[0:3].reshape(3, 1)  # return self.u
            # for 2d
            # self.lastResult = self.u1
            # return self.u1
        except Exception as error:
            logger.info(f"Error in {_module_name}, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    def velocity3d(self, diff_x, diff_y, diff_z):
        vel_x = diff_x / self.dt
        vel_y = diff_y / self.dt
        vel_z = diff_z / self.dt
        self.v13d = np.array([[vel_x], [vel_y], [vel_z]])
        # np.put(self.u, [2, 3], [vel_x, vel_y])
        np.put(self.b3d, [3, 4, 5], [vel_x, vel_y, vel_z])
        pass

