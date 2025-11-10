from setuptools import setup, find_packages

setup(
    name="moead",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.10.7',
        'numpy>=2.3.4',
        'scipy>=1.16.3',
        'pytest>=8.4.2',
    ],
)