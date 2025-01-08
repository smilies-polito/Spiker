from .format_text import indent
from .dict_code import DictCode

class ConstantObj:

	"""
	Declare a VHDL constant

	Methods:
	--------
	code(indent_level = 0)	: generate the string to declare the constant
	"""

	def __init__(self, name : str, const_type : str, value : str):

		"""
		Parameters:
		-----------
		name		: str
			Name of the constant
		const_type	: str
			Type of the constant
		value		: str
			Default value of the constant
		"""

		self.name = name
		self.const_type = const_type
		self.value = value

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare the constant

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return indent(indent_level) + "constant %s : %s := %s;\n" % \
			(self.name, self.const_type, self.value)


class ConstantList(dict):

	"""
	Dictionary of VHDL constants.

	Methods:
	--------
	add(name, const_type, value)	: add a constant object to the dictionary
	code(indent_level = 0)		: generate the string to declare all the
					constants
	"""

	def add(self, name : str, const_type : str, value : str):

		"""
		Add a constant object to the dictionary.

		Parameters:
		-----------
		name		: str
			Name of the constant
		const_type	: str
			Type of the constant
		value		: str
			Value of the constant
		"""

		self[name] = ConstantObj(name, const_type, value)


	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare all the constants

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return DictCode(self, indent_level)
