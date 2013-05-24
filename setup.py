
'''
Configure:
python setup.py install

pyVideoDatasets
Colin Lea
2013
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = []

for e in ext_modules:
	e.pyrex_directives = {
						"boundscheck": False,
						"wraparound": False,
						"infer_types": True
						}
	e.extra_compile_args = ["-w"]

print ext_modules
setup(
	author = 'Colin Lea',
	author_email = 'colincsl@gmail.com',
	description = '',
	license = "FreeBSD",
	version= "0.1",
	name = 'pyVideoDatasets',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include()],
	packages= [	"pyVideoDatasets",
				"pyKinectTools.dataset_readers",
				],
	package_data={'':['*.xml', '*.png', '*.yml', '*.txt']},
	ext_modules = ext_modules
)

