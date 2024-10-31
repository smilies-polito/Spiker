from setuptools import setup, find_packages

setup(
    name							= "spikerplus",
    version							= "2.0.0",
    author							= "Alessio Carpegna",
    author_email					= "alessio.carpegna@polito.it",
    description						= "Build, train, optimize and generate "\
										"hardware accelerators for Spiking "\
										"Neural Networks usign VHDL",
    long_description				= open("README.md").read(),
    long_description_content_type	= "text/markdown",
    url								= "https://github.com/yourusername/"\
										"your_package",
    packages						= find_packages(),
    classifiers						= [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires					= ">=3.9",
    install_requires				= [
        "numpy",
		"torch",
		"snntorch"
    ]
)
