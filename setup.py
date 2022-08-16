from setuptools import find_packages
import os
from setuptools import setup, Command


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
	packages=find_packages(),
	cmdclass={
        'clean': CleanCommand,
    }

)
