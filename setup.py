from setuptools import setup, find_packages

setup(
    name="stochastic-processes-sim",
    version="0.1.0",
    author="Abubakar Mamudu Alutiba",
    author_email="Alutiba04@gmail.com",
    description="Stochastic process simulators for quantitative finance",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pandas>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "jupyter"]
    },
)
