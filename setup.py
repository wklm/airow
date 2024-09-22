from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="training_analytics",
    version="0.1.0",
    author="Wojtek Kulma",
    author_email="wklm@pm.me",
    maintainer="Wojtek Kulma",
    maintainer_email="wklm@pm.me",
    description="Processing and analyzing performance data from activity computers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wklm/training_analytics",
    project_urls={
        "Bug Tracker": "https://github.com/wklm/training_analytics/issues",
        "Changelog": "https://github.com/wklm/training_analytics/blob/master/changelog.md",
        "Homepage": "https://github.com/wklm/training_analytics",
    },
    license="MIT License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "typer",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.11.0",
        "jupyter",
    ],
    extras_require={
        "dev": [
            "coverage",
            "mypy",
            "pytest",
            "ruff",
            "pytest-cov",
            "black",
            "isort",
            "pre-commit",
        ],
    },
)