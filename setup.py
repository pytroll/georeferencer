"""Setup for georeferencer module."""

import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    """A custom setuptools Extension for CMake-based extensions.

    This class represents an extension module that is built using CMake.
    """

    def __init__(self, name):
        """Initializes a CMake extension.

        Args:
            name (str): The name of the extension module.
        """
        super().__init__(name, sources=[])


class CMakeBuildExt(build_ext_orig):
    """A custom build_ext command that integrates CMake into the build process.

    This command:
    - Ensures the required Eigen library is downloaded and available.
    - Determines the Pybind11 CMake directory.
    - Configures and builds the extension using CMake.
    """

    def run(self):
        """Builds the extension using CMake.

        This method performs the following steps:
        1. Ensures the Eigen library is downloaded and available.
        2. Creates a temporary build directory.
        3. Retrieves the Pybind11 CMake directory.
        4. Configures the build using CMake.
        5. Compiles the extension with CMake.

        Raises:
            subprocess.CalledProcessError: If any subprocess command fails.
        """
        eigen_dir = "eigen"
        if not os.path.exists(eigen_dir):
            os.makedirs(eigen_dir, exist_ok=True)
            subprocess.run(
                "curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
                    | tar xz --strip-components=1 -C eigen",
                shell=True,
                check=True,
            )

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        pybind11_dir = subprocess.check_output(
            ["python", "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
            text=True,
        ).strip()

        subprocess.check_call(
            [
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                "-S",
                ".",
                "-B",
                build_temp,
                f"-DPYBIND11_DIR={pybind11_dir}",
            ]
        )

        subprocess.check_call(["cmake", "--build", build_temp])


setup(
    ext_modules=[CMakeExtension("georeferencer.displacement_calc")],
    cmdclass={
        "build_ext": CMakeBuildExt,
    },
)
