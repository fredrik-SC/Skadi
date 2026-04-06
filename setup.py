"""Skadi package setup."""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "PRD.md"

setup(
    name="skadi",
    version="1.0.0",
    description="Standalone offline RF signal identification tool",
    long_description=readme.read_text(encoding="utf-8") if readme.exists() else "",
    long_description_content_type="text/markdown",
    author="Fredrik (SiniusCube Consulting)",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "web": [
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
        ],
    },
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "flask>=3.0.0",
        "flask-socketio>=5.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "skadi=main:main",
        ],
    },
)
