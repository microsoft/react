import setuptools

VERSION = '0.0.1'

setuptools.setup(
    name='react_retrieval',
    author='Haotian Liu',
    author_email='lht@cs.wisc.edu',
    version=VERSION,
    python_requires='>=3.6',
    packages=setuptools.find_packages(exclude=['test', 'test.*']),
    package_data={'': ['resources/*']},
    install_requires=[
        'yacs~=0.1.8',
        'scikit-learn',
        'timm>=0.3.4',
        'numpy>=1.18.0',
        'sharedmem',
        'PyYAML~=5.4.1',
        'Pillow',
    ],
)
