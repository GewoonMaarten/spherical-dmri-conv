import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setuptools.setup(
    name="mudi",
    version="21.07.dev0",
    author="Maarten de Klerk",
    license="MIT",
    description="Concrete autoencoder for sub-sampling multi-dimensional dMRI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GewoonMaarten/geometric-dl-dmri",
    project_urls={
        "Bug Tracker": "https://github.com/GewoonMaarten/geometric-dl-dmri/issues"
    },
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Typing :: Typed",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)
