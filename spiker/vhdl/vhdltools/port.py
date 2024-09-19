from .format_text import indent
from .dict_code import VHDLenum

class PortObj:

	"""
	VHDL port definition.

	Methods:
	--------
	code(indent_level = 0)	: generate the string to declare the port

	"""

	def __init__(self, name : str, direction : str, port_type : str, 
			value : str = ""):

		"""
		Parameters:
		-----------
		name		: str
			Name of the port
		direction	: str
			Direction of the port (in, out, inout, etc.)
		port_type	: str
			Type of the port
		value		: str
			Default value connected to the port
		"""

		self.name = name
		self.direction = direction
		self.port_type = port_type
		self.value = value


	def code(self, indent_level : int = 0) -> str:
		
		"""
		Generate the string to declare the port

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		# No default value
		if (not(self.value) or self.direction == "out"):
			hdl_code = indent(indent_level) + ("%s : %s %s;\n" %
					(self.name, self.direction, self.port_type))

		# Default input value
		elif (self.direction == "in" or self.direction == "inout"):
			hdl_code = indent(indent_level) + ("%s : %s %s := %s;\n"
					% (self.name, self.direction, self.port_type,
						self.value))
		return hdl_code


class PortList(dict):

	"""
	Dictionary of VHDL ports.

	Methods:
	--------
	add(name, gen_type, value)	: add a port object to the dictionary
	code(indent_level = 0)		: generate the string to declare all the
					ports
	"""

	def add(self, name : str, direction : str, port_type : str, 
			value : str = ""):

		"""
		Add a port object to the dictionary.

		Parameters:
		-----------
		name		: str
			Name of the port
		direction	: str
			Direction of the port (in, out, inout, etc.)
		port_type	: str
			Type of the port
		value		: str
			Default value connected to the port
		"""

		self[name] = PortObj(name, direction, port_type, value)


	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare all the ports

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return VHDLenum(self, indent_level)
