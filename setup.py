import os
from setuptools import setup, find_packages, Command


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf build  *.egg-info')


setup(
	name='GP-Optimization',
	version='1.0.0',
	description='GP optimization project',
	author='Bosak Adam',
	author_email='bosadam@seznam.cz',
	url='https://github.com/bosakad/GP-Optimization',
	#packages=find_packages(),
	packages=setuptools.find_namespace_packages(exclude=["tests", "docs","examples","inputs_outputs"]),
	install_requires=[
        "cvxpy>=1.2.1",
        "cycler>=0.11.0",
	"ecos>=2.0.10",
	"fonttools>=4.35.0",
	"kiwisolver>=1.4.4",
	"matplotlib>=3.5.3",
	"Mosek>=9.3.21",
	"networkx>=2.8.5",
	"numpy>=1.23.2",
	"osqp>=0.6.2.post5",
	"packaging>=21.3",
	"Pillow>=9.2.0",
	"pyparsing>=3.0.9",
	"python-dateutil>=2.8.2",
	"qdldl>=0.1.5.post2",
	"scipy>=1.9.0",
	"scs>=3.2.0",
	"six>=1.16.0",
	"tabulate>=0.8.10",
	],
	cmdclass={
        'clean': CleanCommand,
    }

)
