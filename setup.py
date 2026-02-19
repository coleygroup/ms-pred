from setuptools import setup
from setuptools import find_packages
import numpy as np

from Cython.Build import cythonize


# Below is a setup.py call that is failing to correctly compile the cython when i run "python setup.py develop"
# This is because it tries to put the compiled code here 'src/massformer_code/', but that doesn't exist
setup(
    ext_modules=cythonize(
        "src/ms_pred/massformer_pred/massformer_code/algos2.pyx",
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    version="0.0.1",
)
