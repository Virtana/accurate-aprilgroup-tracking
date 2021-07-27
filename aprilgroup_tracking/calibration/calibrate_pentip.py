import numpy as np


class PenTipCalibrator:
    """Estimates the pen tip position c and sphere center c'.

    This Class takes the rotational matrices and translational vectors obtained
    from moving the DodecaPen at a fixed point and obtains the least squares estimate
    of the pen tip position.

    With a fixed tip point Ft, given the poses where tip is also fixed with respect to the
    robot base frame, B, we not have Bt. 
    Then for all i,j poses, we have Rmats * Ft + tvecs = Bt  
    Formulating transform = [Ri, ti] and casting the equation as Ax=b,
    we can obtain a stacked system of equations for some set of indices {(i,j)}_i!=j:
    [Ri - Rj]      [tj - ti]
    [  ...  ] Ft = [  ...  ]
    [  ...  ]      [  ...  ]
    
    We then isolate Ft and estimate Bt as the average over {Rmat_i * Ft + tvec_i}_i.

    Attributes:
        rmats: Rotation matrices obtained from passing rvecs into cv2:Rodrigues.
        tvecs: Translational Vectors obtained from cv2:solvePnP()
    """

    def __init__(self, rmats, tvecs):
        self._rmats = rmats
        self._tvecs = tvecs

    def __call__(self):

        # Ri - Ri+1 for all i in rmats
        A = np.vstack([self._rmats[i] - self._rmats[i+1]
                for i in range(len(self._rmats)-1)])

        # ti+1 - ti for all i in tvecs
        b = np.vstack([self._tvecs[i+1] - self._tvecs[i]
                    for i in range(len(self._tvecs)-1)])

        # Linear least squares estimate of A and b
        fixed_tip = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)
        
        # {Ri @ x + ti}
        rmat_tvecs = []
        for (rmat, tvec) in zip(self._rmats, self._tvecs):
            rmat_tvecs.append(((rmat @ fixed_tip) + tvec.T).reshape(-1))

        # Average of all {Ri @ x + ti}
        sphere_center=np.average(rmat_tvecs, axis=0)

        return fixed_tip, sphere_center
