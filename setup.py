from setuptools import setup, find_packages
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ext_modules = []
cmdclass = {}

try:
    if torch.cuda.is_available():
        ext_modules.append(
            CUDAExtension(
                name="kernel_deltanet_cuda",
                sources=[
                    "kernels/fwht_cuda.cpp",
                    "kernels/in_register_fwht.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": [
                        "-O3",
                        "--use_fast_math",
                        "-lineinfo",
                    ],
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension
except Exception as e:
    print(f"Notice: CUDA extension build skipped ({e}); pure PyTorch/Triton fallback active.")

setup(
    name="kernel-deltanet",
    version="1.0.0",
    author="Anonymous MLSys Authors",
    author_email="anonymous@submission.org",
    description="High-Capacity Linear Recurrence via In-Register Spectral Kernels and Analytical Transmutation",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://anonymous.4open.science/r/kernel-deltanet/",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "triton": ["triton>=2.1.0"],
        "dev": ["pytest", "matplotlib"],
    },
)
