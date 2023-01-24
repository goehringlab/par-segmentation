from setuptools import find_packages, setup

setup(
    name='par_segmentation',
    version='0.1.0',
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
                      'absl-py']
)
