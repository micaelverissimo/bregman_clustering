# Imports
from setuptools import setup, find_packages

# Loading README file
with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name = 'bregmanclustering',
  version = '0.3',
  license='GPL-3.0',
  description = 'A framework to use Bregman Divergence',
  long_description = long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  author = 'Micael Veríssimo de Araújo',
  author_email = 'micael.verissimo@lps.ufrj.br',
  url = 'https://github.com/micaelverissimo/bregmannclustering',
  keywords = ['framework', 'information-geometry', 'machine-learning', 'ai', 'plotting', 'data-visualization'],
  install_requires = [
    'numpy',
    'six>=1.12.0',
    'scipy==1.10.0',
    'future',
    'sklearn',
    'scikit-learn>=0.22.1',
    'matplotlib>=3.1.3',
    'seaborn>=0.10.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)