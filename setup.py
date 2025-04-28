"""Setup for georeferencer module."""

import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    """A custom setuptools Extension for CMake-based extensions."""

    def __init__(self, name):
        """Init."""
        super().__init__(name, sources=[])


class CMakeBuildExt(build_ext_orig):
    """A custom build_ext command that integrates CMake into the build process."""

    def run(self):
        """Run."""
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
                f"-DCMAKE_INSTALL_PREFIX={os.path.abspath(self.build_lib)}",
            ]
        )

        subprocess.check_call(["cmake", "--build", build_temp])
        subprocess.check_call(["cmake", "--install", build_temp])


setup(
    ext_modules=[CMakeExtension("georeferencer.displacement_calc")],
    cmdclass={"build_ext": CMakeBuildExt},
)
