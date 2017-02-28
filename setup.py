from setuptools import setup

setup(name='snape',
    version='0.1',
    description='Snape Realistic Synthetic Dataset Tool',
    url='https://github.com/mbernico/snape',
    author='Mike Bernico',
    author_email='mike.bernico@gmail.com',
    license='Apache 2.0',
    packages=['snape'],
    install_requires=['sklearn',
                      'pandas',
                      'numpy'],
    zip_safe=False)
