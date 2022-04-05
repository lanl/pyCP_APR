"""
2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly,
and to permit others to do so.
"""
from setuptools import setup, find_packages
from glob import glob
__version__ = "1.0.1"

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyCP_APR',
    version=__version__,
    author='Maksim E. Eren, Juston S. Moore, Erik Skau, Manish Bhattarai, Gopinath Chennupati, Boian S. Alexandrov',
    author_email='maksim@lanl.gov',
    description='pyCP_APR: CP-APR Tensor Decomposition with PyTorch Backend.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_dir={'pyCP_APR': 'pyCP_APR'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    setup_requires=[
        'joblib', 'matplotlib', 'numpy', 'numpy-indexed',
        'pandas', 'scikit-learn', 'scipy', 'seaborn', 'torch'
    ],
    url='https://github.com/lanl/pyCP_APR',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8.5',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.8.5',
    install_requires=INSTALL_REQUIRES,
    license='License :: BSD3 License',
)
