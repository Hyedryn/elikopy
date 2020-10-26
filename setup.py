import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="elikopy-qdessain-msimon", # Replace with your own username
    version="0.0.1",
    author="qdessain, msimon",
    author_email="quentin.dessain@student.uclouvain.be, mathieu.simon@student.uclouvain.be",
    description="A set of tools for analysing dMRI",
    url="https://github.com/Hyedryn/python_dti",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'dipy',
        'dmipy',
    ],
)
