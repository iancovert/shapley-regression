import setuptools

setuptools.setup(
    name="shapley-regression",
    version="0.0.1",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="For estimating Shapley values via linear regression.",
    long_description="""
        For calculating the Shapley values of any cooperative game via 
        linear regression. We use an empirically unbiased estimator with
        variance reduction tricks to accelerate convergence.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/iancovert/shapley-regression",
    packages=['shapreg'],
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
