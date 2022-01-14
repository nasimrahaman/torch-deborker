from distutils.core import setup

setup(
    name='torch-deborker',
    version='0.1',
    packages=['torch_db'],
    url='',
    license='MIT',
    author='Nasim Rahaman & Martin Weiss',
    author_email='nasim.rahaman@tuebingen.mpg.de',
    description='Debork all the things! ',
    install_requires = [
        "torch",
        "pytest",
    ]
)
