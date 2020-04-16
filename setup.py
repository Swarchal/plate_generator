from setuptools import setup

setup(
    name="plate_generator",
    author="Scott Warchal",
    packages=["plate_generator"],
    install_requires=["numpy", "scipy"],
    license="MIT",
    tests_require=["pytest"],
)
