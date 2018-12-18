from setuptools import setup, find_packages

setup(
    name='LabelModels',
    version='0.0.1',
    url='https://github.com/BatsResearch/labelmodels.git',
    author='Shiying Luo, Stephen Bach',
    author_email='shiying_luo@brown.edu, sbach@cs.brown.edu',
    description='Lightweight implementations of generative label models for '
                'weakly supervised machine learning',
    packages=find_packages(),
    install_requires=['torch >= 0.4'],
)