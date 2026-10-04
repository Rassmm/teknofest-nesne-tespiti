from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "Teknofest Havacılıkta Yapay Zekâ - Görev 1: Nesne Tespiti"

setup(
    name="teknofest-nesne-tespiti",
    version="1.0.0",
    description="Teknofest Havacılıkta Yapay Zekâ - Görev 1: Nesne Tespiti",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="[Takım Adı]",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
