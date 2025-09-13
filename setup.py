from setuptools import find_packages, setup

setup(
    name="texas_holdem_project",
    version="1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"poker_ai.gui": ["card_images/*.png"]},
    install_requires=[
        "numpy",
        "torch",
        "treys",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "play-poker=poker_ai.cli.play:main",
            "train-poker=poker_ai.cli.train:main",
            "self-play=poker_ai.cli.self_play:main",
        ]
    },
)
