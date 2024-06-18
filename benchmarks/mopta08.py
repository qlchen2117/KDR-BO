
import stat
from typing import Optional
import numpy as np
from platform import machine
import sys
import tempfile
from pathlib import Path
import logging
import urllib
import os
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
logger.addHandler(streamHandler)


class MoptaSoftConstraints:
    """
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    from https://github.com/LeoIV/BAxUS
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
    """

    def __init__(
            self,
            temp_dir: Optional[str] = None,
            binary_path: Optional[str] = None,
            noise_std: Optional[float] = 0,
    ):
        self.dim = 124
        self.lb, self.ub = np.zeros(124), np.ones(124)
        self.noise_std = noise_std
        if binary_path is None:
            self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
            self.machine = machine().lower()
            if self.machine == "armv7l":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_armhf.bin"
            elif self.machine == "x86_64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_elf64.bin"
            elif self.machine == "i386":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_elf32.bin"
            elif self.machine == "amd64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_amd64.exe"
            else:
                raise RuntimeError("Machine with this architecture is not supported")
            self._mopta_exectutable = Path.home().joinpath("dataset/mopta08", self._mopta_exectutable)

            if not self._mopta_exectutable.exists():
                basename = self._mopta_exectutable.name
                logger.info(f"Mopta08 executable for this architecture not locally available. Downloading '{basename}'...")
                urllib.request.urlretrieve(
                    f"https://mopta.papenmeier.io/{self._mopta_exectutable.name}",
                    str(self._mopta_exectutable))
                os.chmod(str(self._mopta_exectutable), stat.S_IXUSR)

        else:
            self._mopta_exectutable = binary_path
        if temp_dir is None:
            self.directory_file_descriptor = tempfile.TemporaryDirectory()
            self.directory_name = self.directory_file_descriptor.name
        else:
            if not os.path.exists(temp_dir):
                logger.warning(f"Given directory '{temp_dir}' does not exist. Creating...")
                os.mkdir(temp_dir)
            self.directory_name = temp_dir

    def __call__(self, x):
        x = np.array(x)
        # if x.ndim == 0:
        #     x = np.expand_dims(x, 0)
        # if x.ndim == 1:
        #     x = np.expand_dims(x, 0)
        # assert x.ndim == 1
        # create tmp dir for mopta binary

        # vals = np.array([self._call(y) for y in x]).squeeze()
        val = self._call(x)
        return val + np.random.normal(0, self.noise_std)


    def _call(self, x: np.ndarray):
        """
        Evaluate Mopta08 benchmark for one point

        Args:
            x: one input configuration

        Returns:value with soft constraints

        """
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None))

    @property
    def optimal_value(self) -> Optional[np.ndarray]:
        """
        Return the "optimal" value.

        Returns:
            np.ndarray: -200, some guessed optimal value we never beat

        """
        return np.array(-200.0)

if __name__ == '__main__':
    from scipy.stats import qmc
    init_py = qmc.Sobol(d=124, scramble=True).random(n=8)
    mopta08 = MoptaSoftConstraints()
    print([mopta08(x) for x in init_py])
