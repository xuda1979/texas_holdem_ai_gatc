from setuptools import setup, find_packages

setup(
    name='texas_holdem_project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'treys',
        'pyyaml',
    ],
    entry_points={'console_scripts': [
        'play-poker=human_vs_ai:main',
        'train-poker=run_training:main',
    ]},
)
