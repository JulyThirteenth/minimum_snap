import numpy as np

class MinJerk:
    """Minimum jerk trajectory planner using closed-form solution.

    This class provides methods to calculate minimum jerk trajectories through waypoints
    with boundary conditions (velocity and acceleration) at start and end points.
    """

    @staticmethod
    def _calc_q_matrix(piece_num: int, times: np.ndarray) -> np.ndarray:
        """Calculates the Q matrix for minimum jerk optimization.

        Args:
            piece_num: Number of trajectory pieces.
            times: Array of time durations for each piece.

        Returns:
            Q matrix for optimization problem.
        """
        Q = np.zeros((6 * piece_num, 6 * piece_num))
        for i in range(piece_num):
            q = np.zeros((6, 6))
            # for j in np.arange(3, 6, 1):
            #     for k in np.arange(3, 6, 1):
            #         q[j, k] = (
            #             (np.math.factorial(j) / np.math.factorial(j - 3))
            #             * (np.math.factorial(k) / np.math.factorial(k - 3))
            #             * (1 / (j + k - 5))
            #             * times[i] ** (j + k - 5)
            #         )
            for j in np.arange(0, 3, 1):
                for k in np.arange(0, 3, 1):
                    q[j, k] = (
                        (np.math.factorial(5 - j) / np.math.factorial(2 - j))
                        * (np.math.factorial(5 - k) / np.math.factorial(2 - k))
                        * (1 / (5 - j - k))
                        * times[i] ** (5 - j - k)
                    )
            Q[i * 6: (i + 1) * 6, i * 6: (i + 1) * 6] = q
        return Q

    @staticmethod
    def _calc_g_matrix(t: float) -> np.ndarray:
        """Calculates the basis function matrix at time t.

        Args:
            t: Time point to evaluate basis functions.

        Returns:
            Matrix containing position, velocity and acceleration basis functions.
        """
        return np.array(
            # [
            #     [1, t, t**2, t**3, t**4, t**5],  # Position
            #     [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],  # Velocity
            #     [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],  # Acceleration
            # ]
            [
                [t**5, t**4, t**3, t**2, t, 1],  # Position
                [5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1, 0],  # Velocity
                [20 * t**3, 12 * t**2, 6 * t, 2, 0, 0]  # Acceleration
            ]
        )

    @staticmethod
    def _calc_m_matrix(piece_num: int, times: np.ndarray) -> np.ndarray:
        """Calculates the mapping matrix from parameters to boundary conditions.

        Args:
            piece_num: Number of trajectory pieces.
            times: Array of time durations for each piece.

        Returns:
            M matrix that maps polynomial coefficients to boundary conditions.
        """
        A = np.zeros((6 * piece_num, 6 * piece_num))
        for i in range(piece_num):
            A[6 * i: 6 * (i + 1), 6 * i: 6 * (i + 1)] = np.vstack(
                [MinJerk._calc_g_matrix(0), MinJerk._calc_g_matrix(times[i])]
            )
        return np.linalg.inv(A)

    @staticmethod
    def _calc_c1_matrix(piece_num: int) -> np.ndarray:
        """Calculates the continuity constraint matrix for position continuity.

        Args:
            piece_num: Number of trajectory pieces.

        Returns:
            C1 matrix enforcing position continuity between pieces.
        """
        C1 = np.zeros((6 * piece_num, 3 * (piece_num + 1)))
        C1[:3, :3] = np.eye(3)  # First piece at time 0
        C1[-3:, -3:] = np.eye(3)  # Last piece at time Tm
        for i in range(0, piece_num - 1):
            C1[3 + 6 * i: 3 + 6 * i + 6, 3 * (i + 1): 3 * (i + 2)] = np.vstack(
                [np.eye(3), np.eye(3)]
            )
        return C1

    @staticmethod
    def _calc_c2_matrix(piece_num: int) -> np.ndarray:
        """Calculates the permutation matrix for free and fixed variables.

        Args:
            piece_num: Number of trajectory pieces.

        Returns:
            C2 matrix that separates fixed and free optimization variables.
        """
        C2 = np.zeros((3 * (piece_num + 1), 3 * (piece_num + 1)))
        C2[:3, :3] = np.eye(3)  # First piece at time 0
        C2[-3:, 3:6] = np.eye(3)  # Last piece at time Tm
        for i in range(0, piece_num - 1):
            C2[3 * i + 3, 6 + i] = 1  # Known position
            C2[3 * i + 4, 6 + piece_num - 1 + 2 * i] = 1  # Unknown velocity
            C2[3 * i + 5, 6 + piece_num - 1 + 2 *
                i + 1] = 1  # Unknown acceleration
        return C2

    @staticmethod
    def gen_poly_traj(waypoints: np.ndarray, times: np.ndarray,
                      vel: np.ndarray, acc: np.ndarray,
                      vel_: np.ndarray, acc_: np.ndarray) -> np.ndarray:
        """Calculates minimum jerk trajectory through given waypoints.

        Args:
            waypoints: Array of waypoints (N x dim) to pass through.
            times: Time durations for each trajectory segment.
            vel: Initial velocity vector.
            acc: Initial acceleration vector.
            vel_: Final velocity vector.
            acc_: Final acceleration vector.

        Returns:
            Array of polynomial coefficients for each trajectory piece (N-1 x 6 x dim).
        """
        piece_num = waypoints.shape[0] - 1
        dim = waypoints.shape[1]
        start = waypoints[0]
        end = waypoints[-1]
        middle = waypoints[1:-1]

        Q = MinJerk._calc_q_matrix(piece_num, times)
        M = MinJerk._calc_m_matrix(piece_num, times)
        C1 = MinJerk._calc_c1_matrix(piece_num)
        C2 = MinJerk._calc_c2_matrix(piece_num)

        R = C2.T @ C1.T @ M.T @ Q @ M @ C1 @ C2
        R_FP = R[: 5 + piece_num, 5 + piece_num:]
        R_PP = R[5 + piece_num:, 5 + piece_num:]

        df = np.vstack([start, vel, acc, end, vel_, acc_, middle])
        dp = -np.linalg.solve(R_PP, R_FP.T @ df)
        d = np.vstack([df, dp])
        p = M @ C1 @ C2 @ d

        return p.reshape(-1, 6, dim)


class MinSnap:
    """Minimum snap trajectory planner using closed-form solution.

    This class provides methods to calculate minimum snap trajectories through waypoints
    with boundary conditions (velocity, acceleration and jerk) at start and end points.
    """

    @staticmethod
    def _calc_q_matrix(piece_num: int, times: np.ndarray) -> np.ndarray:
        """Calculates the Q matrix for minimum snap optimization.

        Args:
            piece_num: Number of trajectory pieces.
            times: Array of time durations for each piece.

        Returns:
            Q matrix for optimization problem.
        """
        Q = np.zeros((8 * piece_num, 8 * piece_num))
        for i in range(piece_num):
            q = np.zeros((8, 8))
            # for j in np.arange(4, 8, 1):
            #     for k in np.arange(4, 8, 1):
            #         q[j, k] = (
            #             (np.math.factorial(j) / np.math.factorial(j - 4))
            #             * (np.math.factorial(k) / np.math.factorial(k - 4))
            #             * (1 / (j + k - 7))
            #             * times[i] ** (j + k - 7)
            #         )
            for j in np.arange(0, 4, 1):
                for k in np.arange(0, 4, 1):
                    q[j, k] = (
                        (np.math.factorial(7 - j) / np.math.factorial(3 - j))
                        * (np.math.factorial(7 - k) / np.math.factorial(3 - k))
                        * (1 / (7 - j - k))
                        * times[i] ** (7 - j - k)
                    )
            Q[i * 8: (i + 1) * 8, i * 8: (i + 1) * 8] = q
        return Q

    @staticmethod
    def _calc_g_matrix(t: float) -> np.ndarray:
        """Calculates the basis function matrix at time t.

        Args:
            t: Time point to evaluate basis functions.

        Returns:
            Matrix containing position, velocity, acceleration and jerk basis functions.
        """
        return np.array(
            # [
            #     [1, t, t**2, t**3, t**4, t**5, t**6, t**7],  # Position
            #     [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t **
            #         4, 6 * t**5, 7 * t**6],  # Velocity
            #     [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3,
            #         30 * t**4, 42 * t**5],  # Acceleration
            #     [0, 0, 0, 6, 24 * t, 60 * t**2, 120 * t**3, 210 * t**4],  # Jerk
            # ]
            [
                [t**7, t**6, t**5, t**4, t**3, t**2, t, 1],  # Position
                [7 * t**6, 6 * t**5, 5 * t**4, 4 * t**3,
                    3 * t**2, 2 * t, 1, 0],  # Velocity
                [42 * t**5, 30 * t**4, 20 * t**3, 12 *
                    t**2, 6 * t, 2, 0, 0],  # Acceleration
                [210 * t**4, 120 * t**3, 60 * t**2, 24 * t, 6, 0, 0, 0]  # Jerk
            ]
        )

    @staticmethod
    def _calc_m_matrix(piece_num: int, times: np.ndarray) -> np.ndarray:
        """Calculates the mapping matrix from parameters to boundary conditions.

        Args:
            piece_num: Number of trajectory pieces.
            times: Array of time durations for each piece.

        Returns:
            M matrix that maps polynomial coefficients to boundary conditions.
        """
        A = np.zeros((8 * piece_num, 8 * piece_num))
        for i in range(piece_num):
            A[8 * i: 8 * (i + 1), 8 * i: 8 * (i + 1)] = np.vstack(
                [MinSnap._calc_g_matrix(0), MinSnap._calc_g_matrix(times[i])]
            )
        return np.linalg.inv(A)

    @staticmethod
    def _calc_c1_matrix(piece_num: int) -> np.ndarray:
        """Calculates the continuity constraint matrix for position continuity.

        Args:
            piece_num: Number of trajectory pieces.

        Returns:
            C1 matrix enforcing position continuity between pieces.
        """
        C1 = np.zeros((8 * piece_num, 4 * (piece_num + 1)))
        C1[:4, :4] = np.eye(4)  # First piece at time 0
        C1[-4:, -4:] = np.eye(4)  # Last piece at time Tm
        for i in range(0, piece_num - 1):
            C1[4 + 8 * i: 4 + 8 * i + 8, 4 * (i + 1): 4 * (i + 2)] = np.vstack(
                [np.eye(4), np.eye(4)]
            )
        return C1

    @staticmethod
    def _calc_c2_matrix(piece_num: int) -> np.ndarray:
        """Calculates the permutation matrix for free and fixed variables.

        Args:
            piece_num: Number of trajectory pieces.

        Returns:
            C2 matrix that separates fixed and free optimization variables.
        """
        C2 = np.zeros((4 * (piece_num + 1), 4 * (piece_num + 1)))
        C2[:4, :4] = np.eye(4)  # First piece at time 0
        C2[-4:, 4:8] = np.eye(4)  # Last piece at time Tm
        for i in range(0, piece_num - 1):
            C2[4 * i + 4, 8 + i] = 1  # Known position
            C2[4 * i + 5, 8 + piece_num - 1 + 3 * i] = 1  # Unknown velocity
            C2[4 * i + 6, 8 + piece_num - 1 + 3 *
                i + 1] = 1  # Unknown acceleration
            C2[4 * i + 7, 8 + piece_num - 1 + 3 * i + 2] = 1  # Unknown jerk
        return C2

    @staticmethod
    def gen_poly_traj(waypoints: np.ndarray, times: np.ndarray,
                      vel: np.ndarray, acc: np.ndarray, jerk: np.ndarray,
                      vel_: np.ndarray, acc_: np.ndarray, jerk_: np.ndarray) -> np.ndarray:
        """Calculates minimum snap trajectory through given waypoints.

        Args:
            waypoints: Array of waypoints (N x dim) to pass through.
            times: Time durations for each trajectory segment.
            vel: Initial velocity vector.
            acc: Initial acceleration vector.
            jerk: Initial jerk vector.
            vel_: Final velocity vector.
            acc_: Final acceleration vector.
            jerk_: Final jerk vector.

        Returns:
            Array of polynomial coefficients for each trajectory piece (N-1 x 8 x dim).
        """
        piece_num = waypoints.shape[0] - 1
        dim = waypoints.shape[1]
        start = waypoints[0]
        end = waypoints[-1]
        middle = waypoints[1:-1]

        Q = MinSnap._calc_q_matrix(piece_num, times)
        M = MinSnap._calc_m_matrix(piece_num, times)
        C1 = MinSnap._calc_c1_matrix(piece_num)
        C2 = MinSnap._calc_c2_matrix(piece_num)

        R = C2.T @ C1.T @ M.T @ Q @ M @ C1 @ C2
        R_FP = R[: 7 + piece_num, 7 + piece_num:]
        R_PP = R[7 + piece_num:, 7 + piece_num:]

        df = np.vstack([start, vel, acc, jerk, end, vel_, acc_, jerk_, middle])
        dp = -np.linalg.solve(R_PP, R_FP.T @ df)
        d = np.vstack([df, dp])
        p = M @ C1 @ C2 @ d

        return p.reshape(-1, 8, dim)


class Polynomial:
    """Represents a any degree and dimension polynomial function.
    coeffs: List of np.poly1d or coefficient arrays, one per dimension.
    Example: [np.poly1d([...]), np.poly1d([...])] for 2D.
    """

    def __init__(self, coeffs):
        self.pos_coeff = [np.poly1d(c) for c in coeffs]
        self.vel_coeff = [np.polyder(c) for c in self.pos_coeff]
        self.acc_coeff = [np.polyder(c) for c in self.vel_coeff]

    def _ensure_array(self, t):
        return np.array([t]) if np.isscalar(t) else np.asarray(t)

    def pos(self, t):
        t = self._ensure_array(t)
        # (dim, len(t))
        return np.array([np.polyval(c, t) for c in self.pos_coeff])

    def vel(self, t):
        t = self._ensure_array(t)
        # (dim, len(t))
        return np.array([np.polyval(c, t) for c in self.vel_coeff])

    def acc(self, t):
        t = self._ensure_array(t)
        # (dim, len(t))
        return np.array([np.polyval(c, t) for c in self.acc_coeff])
    