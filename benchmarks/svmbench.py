import numpy as np
from typing import Optional, Union, List
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path
import urllib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
logger.addHandler(streamHandler)

# NUM_DATA = 10000 # 53500
# TOL = 0.1 # 0.001

class SVMBenchmark:
    def __init__(
            self,
            data_folder: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        """
        From https://github.com/LeoIV/BAxUS
        SVM Benchmark from https://arxiv.org/abs/2103.00349

        Support also a noisy version where the model is trained on random subset of 250 points
        which is used whenever noise_std is greater than 0.

        Args:
            data_folder: the folder where the slice_localization_data.csv is located
            noise_std: noise standard deviation. Anything greater than 0 will lead to a noisy benchmark
            **kwargs:
        """
        self.value = np.inf
        self.best_config = None
        self.noisy = noise_std > 0
        if self.noisy:
            logger.warning("Using a noisy version of SVMBenchmark where training happens on a random subset of 250 points."
                    "However, the exact value of noise_std is ignored.")
        self.dim, self.lb, self.ub, self.noise_std = 388, np.zeros(388), np.ones(388), noise_std
        self.X, self.y = self._load_data(data_folder)
        if not self.noisy:
            np.random.seed(388)
            idxs = np.random.choice(np.arange(len(self.X)), min(10000, len(self.X)), replace=False)
            half = len(idxs) // 2
            self._X_train = self.X[idxs[:half]]
            self._X_test = self.X[idxs[half:]]
            self._y_train = self.y[idxs[:half]]
            self._y_test = self.y[idxs[half:]]

    def _load_data(self, data_folder: Optional[str] = None):
        if data_folder is None:
            data_folder = Path.home().joinpath("dataset/svm/data")
        if not data_folder.joinpath("CT_slice_X.npy").exists():
            sld_dir = data_folder.joinpath("slice_localization_data.csv.xz")
            sld_bn = sld_dir.name
            logger.info(f"Slice localization data not locally available. Downloading '{sld_bn}'...")
            urllib.request.urlretrieve(
                f"http://mopta-executables.s3-website.eu-north-1.amazonaws.com/{sld_bn}",
                sld_dir)
            data = pd.read_csv(
                data_folder.joinpath("slice_localization_data.csv.xz")
            ).to_numpy()
            X = data[:, :385]
            y = data[:, -1]
            np.save(data_folder.joinpath("CT_slice_X.npy"), X)
            np.save(data_folder.joinpath("CT_slice_y.npy"), y)
        X = np.load(data_folder.joinpath("CT_slice_X.npy"))
        y = np.load(data_folder.joinpath("CT_slice_y.npy"))
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        return X, y

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        x = np.array(x)
        assert x.ndim == 1
        y = x ** 2


        C = 0.01 * (500 ** y[387])
        gamma = 0.1 * (30 ** y[386])
        epsilon = 0.01 * (100 ** y[385])
        length_scales = np.exp(4 * y[:385] - 2)

        # svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=TOL)
        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, max_iter=200)
        if self.noisy:
            np.random.seed(None)
            idxs = np.random.choice(np.arange(len(self.X)), min(500, len(self.X)), replace=False)
            half = len(idxs) // 2
            X_train = self.X[idxs[:half]]
            X_test = self.X[idxs[half:]]
            y_train = self.y[idxs[:half]]
            y_test = self.y[idxs[half:]]
            svr.fit(X_train / length_scales, y_train)
            pred = svr.predict(X_test / length_scales)
            error = np.sqrt(np.mean(np.square(pred - y_test)))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                svr.fit(self._X_train / length_scales, self._y_train)
            pred = svr.predict(self._X_test / length_scales)
            error = np.sqrt(np.mean(np.square(pred - self._y_test)))

        if error < self.value:
            self.best_config = np.log(y)
            self.value = error
        return error.item()

if __name__ == '__main__':
    from scipy.stats import qmc
    svmBench = SVMBenchmark()
    init_py = qmc.Sobol(d=svmBench.dim, scramble=True).random(n=8)
    import time
    t1 = time.monotonic()
    print([svmBench(x) for x in init_py])
    t2 = time.monotonic()
    print(f'{t2-t1}s')