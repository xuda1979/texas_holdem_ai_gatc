from setuptools import find_packages, setup

setup(
    name='texas_holdem_project',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'treys',
        'pyyaml',
    ],
    entry_points={'console_scripts': [
        'play-poker=poker_ai.cli.play:main',
        'train-poker=poker_ai.cli.train:main',
        'self-play=poker_ai.cli.self_play:main',
    ]},
)
