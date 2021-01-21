from setuptools import find_packages, setup

setup(
    name='chemprop-IR',
    version='0.0.1',
    author='Charles McGill, Michael Forsuelo, Yanfei Guan, William Green',
    author_email='cjmcgill@mit.edu, forsuelo@mit.edu, whgreen@mit.edu',
    description='IR spectra prediction Prediction with Message Passing Neural Networks',
    url='https://github.com/gfm-collab/chemprop-IR',
    license='MIT',
    packages=find_packages(),
    keywords=['chemistry', 'machine learning', 'property prediction', 'message passing neural network']
)
