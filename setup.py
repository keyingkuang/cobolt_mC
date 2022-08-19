from setuptools import setup, find_packages

setup(
    name='cobolt',
    version='0.0.1',
    author='boyinggong',
    author_email='boyinggong@berkeley.edu',
    description='A package for joint analysis of multimodal single-cell sequencing data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boyinggong/cobolt",
    project_urls={
        "Bug Tracker": "https://github.com/boyinggong/cobolt/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'torch',
        'umap-learn',
        'python-igraph',
        'sklearn',
        'xgboost',
        'seaborn',
        'leidenalg',
    ],
    python_required=">=3.7"
)