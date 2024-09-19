from .format_text import indent
from .dict_code import VHDLenum

class GenericObj:

	"""
	VHDL generic definition.

	Methods:
	--------
	code(indent_level = 0)	: generate the string to declare the generic
	"""

	def __init__(self, name : str, gen_type : str, value : str = ""):

		"""
		Parameters:
		-----------
		name		: str
			Name of the generic variable
		gen_type	: str
			Type of the generic variable
		value		: str
			Default value of the generic variable
		"""

		self.name = name
		self.gen_type = gen_type
		self.value = value

	def code(self, indent_level : int = 0):

		"""
		Generate the string to declare the generic

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		if self.value:

			# Assign a default value to the generic
			hdl_code = indent(indent_level) + ("%s : %s := %s;\n" %
					(self.name, self.gen_type, self.value))

		else:

			# No default value
			hdl_code = indent(indent_level) + ("%s : %s;\n" %
					(self.name, self.gen_type))

		return hdl_code


class GenericList(dict):

	"""
	Dictionary of generic variables.

	Methods:
	--------
	add(name, gen_type, value)	: add a generic object to the dictionary
	code(indent_level = 0)		: generate the string to declare all the
					generics
	"""

	def add(self, name : str, gen_type : str, value : str = ""):

		"""
		Add a generic object to the dictionary.

		Parameters:
		-----------
		name		: str
			Name of the generic variable
		gen_type	: str
			Type of the generic variable
		value		: str
			Default value of the generic variable
		"""

		self[name] = GenericObj(name, gen_type, value)


	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare all the generics

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return VHDLenum(self, indent_level)
