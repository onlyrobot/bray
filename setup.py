from setuptools import setup, find_packages

setup(
    name='bray',
    version='1.0.0',
    description='Bray for based on ray and better than ray',
    author='Peng Yao',
    author_email='onlyrobot@qq.com',
    url='',
    install_requires=[
        'numpy',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'bray = bray.launch:main'
        ]
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)