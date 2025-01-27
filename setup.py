import os
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools.command.install import install

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuildExt(build_ext_orig):
    def run(self):

        eigen_dir = "eigen"
        if not os.path.exists(eigen_dir):
            print("Preparing Eigen...")
            os.makedirs(eigen_dir, exist_ok=True)
            subprocess.run(
                "curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz --strip-components=1 -C eigen",
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
                "-S", ".",
                "-B", build_temp,
                f"-DPYBIND11_DIR={pybind11_dir}",
            ]
        )

        subprocess.check_call(["cmake", "--build", build_temp])

setup(
    name="georeferencer",
    version="0.0.1",
    author="Jacob Nilsson",
    author_email="jacob.nilsson@smhi.se",
    description="Python package for georeferencing satellite imagery.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    install_requires=open("requirements.txt").read().splitlines(),
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    ext_modules=[CMakeExtension('georeferencer.displacement_calc')],
    cmdclass={
        "build_ext": CMakeBuildExt,
    },
)
