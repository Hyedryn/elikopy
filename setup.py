import setuptools

import io

try:
    with io.open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except (IOError, OSError) as e:
    print(e.errno)

setuptools.setup(
    name="elikopy",
    packages = ['elikopy'],
    license='agpl-3.0',
    version="0.2",
    author="qdessain, msimon",
    author_email="quentin.dessain@student.uclouvain.be, mathieu.simon@student.uclouvain.be",
    description="A set of tools for analysing dMRI",
    url="https://github.com/Hyedryn/elikopy",
    download_url = 'https://github.com/Hyedryn/elikopy/archive/refs/tags/v0.2.tar.gz',
    #packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'dipy',
        'dmipy',
        'lxml',
        'PyPDF2',
        'fpdf',
        'matplotlib',
        'cvxpy'
    ],
)
