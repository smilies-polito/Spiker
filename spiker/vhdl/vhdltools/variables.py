from .format_text import indent
from .dict_code import DictCode

class VariableObj:

	"""
	Declare a VHDL variable.

	Methods:
	--------
	code(indent_level = 0)	: generate the string to declare the variable
	"""

	def __init__(self, name : str, var_type : str, value : str = ""):

		"""
		Parameters:
		-----------
		name		: str
			Name of the variable
		var_type	: str
			Type of the variable
		*args		: str
			List of default values of the variable. In
			practice only the first is used.
		"""

		self.name = name
		self.var_type = var_type
		self.value = value


	def code(self, indent_level : int = 0):

		"""
		Generate the string to declare the variable

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		if self.value:
			return indent(indent_level) + \
					("variable %s : %s := %s;\n" \
					% (self.name, self.var_type, \
					self.value))
		else:
			return indent(indent_level) + ("variable %s : %s;\n" % \
					(self.name, self.var_type))


class VariableList(dict):

	"""
	Dictionary of VHDL variables.

	Methods:
	--------
	add(name, var_type, *args)	: add a variable object to the
					dictionary
	code(indent_level = 0)		: generate the string to declare all the
					variabeles
	"""

	def add(self, name, var_type, value = ""):

		"""
		Add a variable object to the dictionary.

		Parameters:
		-----------
		name		: str
			Name of the variable
		var_type	: str
			Type of the variable
		*args		: str
			List of default values of the variable. In
			practice only the first is used.
		"""

		self[name] = VariableObj(name, var_type, value)

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare all the variables

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return DictCode(self, indent_level)
