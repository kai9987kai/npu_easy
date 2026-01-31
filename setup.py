from setuptools import setup, find_packages

setup(
    name="npu-easy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[], # Zero hard dependencies
    extras_require={
        "intel": ["onnxruntime-openvino"],
        "amd": ["onnxruntime-directml"],
        "qualcomm": ["onnxruntime"], # Requires separate QNN SDK setup
    },
    author="Antigravity",
    description="A zero-dependency wrapper for easy NPU usage in Python on Windows.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ],
)
