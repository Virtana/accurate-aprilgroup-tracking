import numpy as np
from typing import Tuple


class PenTipCalibrator:
    """Estimates the pen tip position c and sphere center c'.

    This Class takes the rotational matrices and translational vectors obtained
    from moving the DodecaPen at a fixed point and obtains the
    least squares estimate of the pen tip position.

    With a fixed tip point Ft, given the poses where tip is also
    fixed with respect to the robot base frame, B, we not have Bt.
    Then for all i,j poses, we have Rmats * Ft + tvecs = Bt
    Formulating transform = [Ri, ti] and casting the equation as Ax=b,
    we can obtain a stacked system of equations for some
    set of indices {(i,j)}_i!=j:
    [Ri - Rj]      [tj - ti]
    [  ...  ] Ft = [  ...  ]
    [  ...  ]      [  ...  ]

    We then isolate Ft and estimate Bt as the average over
    {Rmat_i * Ft + tvec_i}_i.

    Attributes:
        rmats: Rotation matrices obtained from passing
                rvecs into cv2:Rodrigues.
        tvecs: Translational Vectors obtained from cv2:solvePnP()
    """

    def __init__(self, rmats, tvecs):
        self._rmats = rmats
        self._tvecs = tvecs

    def _algebraic_two_step(self) -> Tuple[np.ndarray, np.ndarray]:

        # [Ri - Ri+1] for all i in rmats
        A = np.vstack([self._rmats[i] - self._rmats[i+1]
                    for i in range(len(self._rmats)-1)])

        # [ti+1 - ti] for all i in tvecs
        dbfps = [(self._tvecs[i+1].T - self._tvecs[i].T).reshape((3, 1))
                for i in range(len(self._tvecs)-1)]
        b = np.vstack(dbfps)

        # Linear least squares estimate of A and b
        fixed_tip = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)

        # {Ri @ x + ti}
        rmat_c_tvecs = []
        for (rmat, tvec) in zip(self._rmats, self._tvecs):
            rmat_c_tvecs.append(((rmat @ fixed_tip) + tvec.T).reshape(-1))

        # Average of all {Ri @ x + ti}
        sphere_center = np.average(rmat_c_tvecs, axis=0)

        return fixed_tip, sphere_center

    def _algebraic_one_step(self) -> Tuple[np.ndarray, np.ndarray]:

        # Create arrays which match the dims and shapes
        # expected by linalg.lstsq
        t_i_shaped = np.array(self._tvecs).reshape(-1, 1)
        print("t i shape", t_i_shaped)

        r_i_shaped = []
        for r in self._rmats:
            r_i_shaped.extend(np.concatenate((r, -np.identity(3)), axis=1))
        r_i_shaped = np.stack(r_i_shaped)

        # Run least-squares, extract the positions
        lstsq = np.linalg.lstsq(r_i_shaped, -t_i_shaped)
        fixed_tip = lstsq[0][0:3]
        sphere_center = lstsq[0][3:6]

        # Calculate the 3D residual RMS error
        residual_vectors = np.array((
            t_i_shaped + r_i_shaped@lstsq[0]).reshape(len(self._tvecs), 3))
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        residual_rms = np.sqrt(np.mean(residual_norms ** 2))

        return fixed_tip, sphere_center
