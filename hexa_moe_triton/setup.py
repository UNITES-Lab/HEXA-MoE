"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os, sys
import subprocess
import platform as pf

from typing import List, Tuple

from setuptools import setup, find_packages, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

try:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
except:
    IS_HIP_EXTENSION = False

if len(sys.argv) <= 1:
    sys.argv += ['install']

root_path = os.path.dirname(sys.argv[0])
root_path = root_path if root_path else '.'
root_path = os.path.abspath(root_path)

os.chdir(root_path)

class Tester(Command):
    """Cmdclass for `python setup.py test`.
    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'test the code using pytest'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Run pytest."""
        subprocess.check_call('python3 -m pytest -v -s tests/', shell=True)
        
def install(use_cuda, use_nccl):
    ext_libs = []
    if pf.system() == 'Linux':
        ext_args = ['-Wno-sign-compare', '-Wno-unused-but-set-variable', '-Wno-terminate', '-Wno-unused-function', '-Wno-strict-aliasing']
    elif pf.system() == 'Darwin':
        ext_args = ['-mmacosx-version-min=10.13']
    else:
        ext_args = []

    if not use_cuda:
        use_nccl = False
        extension = CppExtension
    else:
        ext_libs += ['cuda', 'nvrtc'] if not IS_HIP_EXTENSION else []
        ext_args += ['-DUSE_GPU']
        extension = CUDAExtension

    if use_nccl:
        if IS_HIP_EXTENSION:
            ext_libs += ['rccl']
        else:
            ext_libs += ['nccl']
        ext_args += ['-DUSE_NCCL']

    setup(
        name='hexa_moe',
        version='0.1',
        description='A Refactored and Efficient Mixture-of-Experts Implementation.',
        license='MIT',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Environment :: GPU',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords=['Mixture of Experts', 'MoE', 'Optimization'],
        packages=find_packages(exclude=['tests']),
        python_requires='>=3.6, <4',
        install_requires=[
            "numpy",
        ],
        zip_safe=False,
        extras_require={
            'test': [
                'GPUtil>=1.4.0',
                'pytest-subtests>=0.4.0',
                'pytest>=6.2.2',
            ],
        },
        cmdclass={
            "build_ext": BuildExtension,
            'test': Tester
        }
    )
    
if int(os.environ.get('NO_CUDA', 0)) == 1:
    print('Installing without CUDA extension..')
    install(use_cuda=False, use_nccl=False)
else:
    try:
        install(use_cuda=True, use_nccl=True)
    except:
        print('Try installing without NCCL extension..')
        try:
            install(use_cuda=True, use_nccl=False)
        except:
            print('Try installing without CUDA extension..')
            install(use_cuda=False, use_nccl=False)