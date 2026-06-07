from pathlib import Path
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
VERSION_SOURCE = (ROOT / "npu_easy" / "_version.py").read_text(encoding="utf-8")
VERSION = re.search(r'__version__ = "([^"]+)"', VERSION_SOURCE).group(1)


setup(
    name="npu-easy",
    version=VERSION,
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "cpu": ["onnxruntime"],
        "intel": ["onnxruntime-openvino"],
        "qualcomm": ["onnxruntime-qnn"],
        "directml": ["onnxruntime-directml"],
        "amd": ["onnxruntime-directml"],
        "nvidia": ["onnxruntime-gpu"],
        "dev": ["build", "numpy", "pytest>=8", "ruff"],
    },
    python_requires=">=3.9",
    author="Antigravity",
    description=(
        "Zero-hard-dependency ONNX Runtime provider selection, diagnostics, "
        "and benchmarking for NPUs, GPUs, and CPUs."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/kai9987kai/npu_easy",
        "Issues": "https://github.com/kai9987kai/npu_easy/issues",
    },
    entry_points={
        "console_scripts": [
            "npu-easy=npu_easy.__main__:main",
        ]
    },
)
