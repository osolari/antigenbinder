from setuptools import setup, find_packages
import numpy

setup(
    name="antigenbinder",
    version="0.0.1",
    author="Omid Shams Solari",
    author_email="solari@berkeley.edu",
    maintainer="Omid Shams Solari",
    maintainer_email="solari@berkeley.edu",
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        "setuptools>=18.0",
        "cython",
    ],
    tests_require=["pytest"],
    packages=find_packages(),
    include_package_data=True,
    include_dirs=[numpy.get_include()],
)
