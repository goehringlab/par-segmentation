from setuptools import find_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='par_segmentation',
    version='0.1.4',
    license="CC BY 4.0",
    author='Tom Bland',
    author_email='tom_bland@hotmail.co.uk',
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'ipywidgets',
                      'scikit-image',
                      'jupyter',
                      'opencv-python',
                      'joblib',
                      'tensorflow>=2.9.1',
                      'tqdm',
                      'pandas',
                      'absl-py'],
    description='Cell cortex segmentation in C. elegans PAR protein images',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
