import os
import sys
from typing import List

from setuptools import find_packages, setup

# copied from pytorch-lightning
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRE = os.path.join(_PATH_ROOT, 'requirements')


def _load_requirements(path_dir: str, file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    """Load requirements from a file
    >>> _load_requirements(_PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install pytorch-lightning[dev, docs]`
# From local copy of repo, use like `pip install ".[dev, docs]"`
extras = {
    'examples': _load_requirements(path_dir=_PATH_REQUIRE, file_name='examples.txt'),
    'test': _load_requirements(path_dir=_PATH_REQUIRE, file_name='test.txt')
}
extras['dev'] = extras['examples'] + extras['test']
extras['all'] = extras['dev']


setup(
    name='xynn',
    version='0.1',
    description='A collection of Tabular NN models with a scikit-learn api',
    url='https://github.com/jrfiedler/shim_temp',
    author='James Fiedler',
    author_email='jrfiedler@gmail.com',
    license='MIT',
    python_requires=">=3.7",
    packages=find_packages(exclude=['tests','tests/*',]),
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    setup_requires=[],
    install_requires=_load_requirements(_PATH_ROOT),
    extras_require=extras,
    )