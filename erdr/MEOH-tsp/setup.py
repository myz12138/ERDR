from setuptools import setup, find_packages

setup(
    name='llm4ad',
    version='1.0',
    description='Large Language Model for Algorithm Design Platform ',
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires='>=3.9,<3.13',
    install_requires=[
        'numpy<2',
        'torch',
        'tensorboardX',
        'wandb',
        'scipy',
        'tqdm',
        'numba',
        'requests',
        'pytz',
        'matplotlib',
        'pyyaml',
        'python-docx',
        'ttkbootstrap'
    ]
)
