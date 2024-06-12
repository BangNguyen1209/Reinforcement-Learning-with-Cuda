from setuptools import setup, find_packages

setup(
    name='lux',
    version='0.1.0',
    author='Khanh.VuDinhDuy',
    author_email='duykhanh100902@gmail.com',
    packages=find_packages(exclude=['tests*']),
    description='Matching python environment code for Lux AI 2021 Kaggle competition and a gym interface for RL models',
    long_description=open('README.md').read(),
    install_requires=[
        "pytest",
        "stable_baselines3==1.2.1a2",
        "numpy",
        "tensorboard",
        "gym==0.19.0"
    ],
    package_data={'lux': ['game/game_constants.json', 'env/rng/rng.js', 'env/rng/seedrandom.js']},
    test_suite='nose2.collector.collector',
    tests_require=['nose2'],
)