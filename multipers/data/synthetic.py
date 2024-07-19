import numpy as np


def noisy_annulus(
    n1: int = 1000,
    n2: int = 200,
    r1: float = 1,
    r2: float = 2,
    dim: int = 2,
    center: np.ndarray | list | None = None,
    **kwargs
) -> np.ndarray:
    """Generates a noisy annulus dataset.

    Parameters
    ----------
    r1 : float.
            Lower radius of the annulus.
    r2 : float.
            Upper radius of the annulus.
    n1 : int
            Number of points in the annulus.
    n2 : int
            Number of points in the square.
    dim : int
            Dimension of the annulus.
    center: list or array
            center of the annulus.

    Returns
    -------
    numpy array
            Dataset. size : (n1+n2) x dim

    """
    theta = np.random.normal(size=(n1, dim))
    theta /= np.linalg.norm(theta, axis=1)[:, None]
    rs = np.sqrt(np.random.uniform(low=r1**2, high=r2**2, size=n1))
    annulus = rs[:, None] * theta
    if center is not None:
        annulus += np.array(center)
    diffuse_noise = np.random.uniform(size=(n2, dim), low=-1.1 * r2, high=1.1 * r2)
    if center is not None:
        diffuse_noise += np.array(center)
    return np.vstack([annulus, diffuse_noise])


def three_annulus(num_pts: int = 500, num_outliers: int = 500):
    X = np.block(
        [
            [np.random.uniform(low=-2, high=2, size=(num_outliers, 2))],
            [
                np.array(
                    noisy_annulus(
                        r1=0.6,
                        r2=0.9,
                        n1=(int)(num_pts * 1 / 3),
                        n2=0,
                        center=[1, -0.2],
                    )
                )
            ],
            [
                np.array(
                    noisy_annulus(
                        r1=0.4,
                        r2=0.55,
                        n1=(int)(num_pts * 1 / 3),
                        n2=0,
                        center=[-1.2, -1],
                    )
                )
            ],
            [
                np.array(
                    noisy_annulus(
                        r1=0.3,
                        r2=0.4,
                        n1=(int)(num_pts * 1 / 3),
                        n2=0,
                        center=[-0.7, 1.1],
                    )
                )
            ],
        ]
    )
    return X


def orbit(n: int = 1000, r: float = 1.0, x0=[]):
    point_list = []
    if len(x0) != 2:
        x, y = np.random.uniform(size=2)
    else:
        x, y = x0
    point_list.append([x, y])
    for _ in range(n - 1):
        x = (x + r * y * (1 - y)) % 1
        y = (y + r * x * (1 - x)) % 1
        point_list.append([x, y])
    return np.asarray(point_list, dtype=float)


def get_orbit5k(num_pts=1000, num_data=5000):
    from sklearn.preprocessing import LabelEncoder

    rs = [2.5, 3.5, 4, 4.1, 4.3]
    labels = np.random.choice(rs, size=num_data, replace=True)
    X = [orbit(n=num_pts, r=r) for r in labels]
    labels = LabelEncoder().fit_transform(labels)
    return X, labels
