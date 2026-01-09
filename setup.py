from setuptools import setup, find_packages

__package_name__ = "uniself"


def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


__version__, cmdclass = get_version_and_cmdclass(__package_name__)

if __version__ in (None, "", "unknown", "UNKNOWN"):
    __version__ = "1.2.0"


# noinspection PyTypeChecker
setup(
    name=__package_name__,
    version=__version__,
    description="UNISELF: A Unified Network with Instance Normalization and Self-Ensembled Lesion Fusion for Multiple Sclerosis Lesion Segmentation",
    long_description="UNISELF: A Unified Network with Instance Normalization and Self-Ensembled Lesion Fusion for Multiple Sclerosis Lesion Segmentation",
    author="Jinwei Zhang",
    author_email="jwzhang@jhu.edu",
    url="https://github.com/Jinwei1209/uniself",
    license="Apache License, 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    keywords="ms lesion segmentation",
    entry_points={
        "console_scripts": [
            "uniself-msseg=uniself.test:main",
        ]
    },
    install_requires=[
        "nibabel",
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "tqdm",
        "torchio",
        "scikit-image"
    ],
    cmdclass=cmdclass,
)