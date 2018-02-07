from setuptools import setup, find_packages

setup(
    name='klearn',
    version='0.0.1',
    description='''Sci-Kit Learn Add-on''',
    author='Kevin Liao',
    author_email='kevin.lwk.liao@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas==0.20.1',
        'scikit-learn==0.19.1',
        # 'seaborn==0.8.1',
        # 'plotly==2.2.3',
        # 'xgboost==0.6a2'
    ],
    dependency_links=[],
    platforms='any'
)
