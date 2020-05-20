
from setuptools import setup

setup(name='snape',
      version='0.2',  # TODO: this needs to be controlled w a tag
      description='Snape Realistic Synthetic Dataset Tool',
      url='https://github.com/mbernico/snape',
      author='Mike Bernico',
      author_email='mike.bernico@gmail.com',
      license='Apache 2.0',
      packages=['snape'],
      install_requires=['scikit-learn>=0.20',
                        'pandas',
                        'numpy',
                        'requests',
                        'beautifulsoup4',
                        'lxml'
                        ],
      zip_safe=False)
