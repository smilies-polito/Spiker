from .format_text import indent
from .dict_code import DictCode

class PackageObj:

	"""
	VHDL packages to use.

	Methods:
	--------
	code(libname = "ieee")	: generate the string to use the specific
				package
	"""

	def __init__(self, name : str, *args : str):

		"""
		Parameters:
		-----------
		name	: str
			Package name
		*args	: str, optional
			Arbitrary number of strings, representing which part of
			the package to use. If not specified "all" is used.
		"""

		self.source = "File Location Unknown."
		self.name = name

		if args:
			# In practice only the first operator is considered
			self.operator = args[0]
		else:
			self.operator = "all"



	def code(self, libname : str = "ieee", indent_level : int = 0) -> str:

		"""
		Generate the string of text to use the selected package
		subportion.

		Parameters:
		-----------
		libname	: str
			Name of the library from which the package will be taken
			(e.g. ieee)
		indent_level	: int, optional
			Number of indentations to insert between printed
			elements.
		"""

		hdl_code = ""

		hdl_code = hdl_code + indent(indent_level) + ("use "
			"%s.%s.%s;\n" % (libname, self.name,
			self.operator))

		return hdl_code


class PackageList(dict):

	"""
	Create a dictionary of packages. key = name of the package, value =
	package object

	Methods:
	--------
	add(name, *args)	: add a package object to the dictionary
	code(libname = "work")	: generate the code to use all the packages

	"""

	def add(self, name : str, *args : str):

		"""
		Add a package object to the dictionary.

		Parameters:
		-----------
		name	: str
			Name of the package
		args	: str
			
		"""

		self[name] = PackageObj(name, *args)


	def code(self, libname : str = "ieee", indent_level : int = 0) -> str:

		"""
		Generate the code to use all the packages.

		Parameters:
		----------
		libname	: str, optional
			Name of the library from which to take all the packages
		indent_level	: int, optional
			Number of indentations to insert between printed
			elements.
		"""

		hdl_code = ""

		for eachPkg in self:
			hdl_code = hdl_code + self[eachPkg].code(libname,
					indent_level)

		return hdl_code


class ContextObj:

	"""
	VHDL context.
	"""

	def __init__(self, name : str):

		"""
		Parameters:
		-----------
		name	: str
			Name of the VHDL context
		"""

		self.source = "File Location Unknown."
		self.name = name


class ContextList(dict):

	"""
	Dictionary of VHDL contexts.

	Methods:
	--------
	add(name)	: add a context object to the dictionary
	"""

	def add(self, name : str):

		"""
		Add a context object to the dictionary.

		Parameters:
		-----------
		name	: str
			Name of the VHDL context
		"""

		self[name] = ContextObj(name)


class LibraryObj:

	"""
	Full library object. Allow to specify the library and all the packages
	that will be used from this library.

	Methods:
	--------
	code(indent_level = 0)	: generate the full code to import the library
				and use the selected packages
	"""

	def __init__(self, name : str = "ieee"):

		"""
		Parameters:
		-----------
		name	: str
			Name of the VHDL library

		"""

		self.name = name
		self.package = PackageList()
		self.context = ContextList()

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the full code to import the library and use the
		selected packages.

		Parameters:
		-----------
		indent_level	: int, optional
			Number of indentations to insert between printed
			elements.
		"""

		hdl_code = ""

		# Declare the library
		hdl_code = hdl_code + indent(indent_level + 0) + ("library %s;\n" % self.name)

		# Declare the context
		for j in self.context:
			hdl_code = hdl_code + indent(indent_level + 1) + ("context %s.%s;\n" % (self.name, self.context[j].name))

		# Import all the desired packages
		hdl_code = hdl_code + self.package.code(self.name)

		return hdl_code


class LibraryList(dict):

	"""
	Dictionary of libraries, together with their packages

	Methods:
	--------
	add(name)		: add a library object to the dictionary
	code(indent_level = 0)	: generate the full code to import all the
				libraries and use the corresponding packages.

	"""

	def add(self, name : str):

		"""
		Add a library object to the dictionary.

		Parameters:
		-----------
		name	: str
			Name of the VHDL library to use.
		"""

		self[name] = LibraryObj(name)

	def code(self, indent_level : int = 0):

		"""
		Generate the full code to import all the libraries and use the
		corresponding packages.
		
		Parameters:
		-----------
		indent_level	: int, optional
			Number of indentations to insert between printed
			elements.

		"""

		return DictCode(self) + "\n"
