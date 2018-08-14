from setuptools import setup

setup(
    name='keras-global-self-attention',
    version='0.0.1',
    packages=['keras_global_self_attention'],
    url='https://github.com/PoWWoP/keras-global-self-attention',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
