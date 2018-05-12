from setuptools import setup
from setuptools import find_packages

setup(
    name='klearn',
    version='0.0.1',
    description='''Data Science and Machine Learning Tool Kits''',
    author='Kevin Liao',
    author_email='kevin.lwk.liao@gmail.com',
    license='MIT',
    install_requires=[
        'numpy>=1.9.1',
        'scipy>=0.14',
        'pandas>=0.20.1',
        'scikit-learn>=0.19.1',
        'statsmodels>=0.8.0',
        # 'xgboost==0.7.post4'
    ],
    extras_require={
        'visualize': [
            'seaborn>=0.8.1',
            'plotly>=2.2.3',
            ],
    },
    dependency_links=[],
    platforms='any',
    packages=find_packages(),
)
