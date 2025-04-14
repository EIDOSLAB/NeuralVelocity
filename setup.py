from setuptools import setup, find_packages

setup(
    name="src",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
