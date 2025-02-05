from .format_text import indent
from .text import GenericCodeBlock
from .dict_code import DictCode

class For:
	
	"""
	VHDL for loop.

	Methods:
	--------
	code(indent_level = 0)	: generate the string of the for loop
	"""

	def __init__(self, name : str = "", start : int = 0, stop : int = 1,
			iter_name : str= "i", direction : str= "up", 
			loop_type = "loop"):

		"""
		Parameters:
		-----------
		name		: str, optional
			Name of the loop.
		start		: int, optional
			Starting point of the loop
		stop		: int, optional
			Ending point of the loop
		iter_name	: str, optional
			Name of the variable used to iterate
		direction	: str, optional
			Direction of the loop. Can be "up" or "down"
		"""

		self.name = name
		self.start = start
		self.stop = stop
		self.iter_name = iter_name
		self.direction = direction
		self.body = GenericCodeBlock()
		self.loop_type = loop_type

	def code(self, indent_level = 0):

		"""
		Generate the for loop string

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		hdl_code = ""


		if self.body:

			# The loop has a name
			if(self.name == ""):
				hdl_code = hdl_code + indent(indent_level) + \
						"for " + self.iter_name + " in "

			# The loop has not a name
			else:
				hdl_code = hdl_code + indent(indent_level) + \
						self.name + " : for " + \
						self.iter_name + " in "

			# From higher to lower index
			if(self.direction == "down"):
				hdl_code = hdl_code + str(self.start) + \
						" downto " + str(self.stop) + \
						"\n"

			# From lower to higher index
			else:
				hdl_code = hdl_code + str(self.start) + " to " \
						+ str(self.stop) + "\n"

			

			if self.loop_type == "loop":
				hdl_code = hdl_code + indent(indent_level) + \
						"loop\n" 
			else:
				hdl_code = hdl_code + indent(indent_level) + \
						"generate\n" 

			hdl_code = hdl_code + self.body.code(indent_level + 1) \
					+ "\n"

			if self.loop_type == "loop":
				hdl_code = hdl_code + indent(indent_level) + \
					"end loop " + self.name + ";\n\n"
			else:
				hdl_code = hdl_code + indent(indent_level) + \
					"end generate " + self.name + ";\n\n"

		return hdl_code


class ForList(dict):

	def __init__(self):
		self.index = 0

	def add(self, name : str = "", start : int = 0, stop : int = 1,
			iter_name : str= "i", direction : str= "up"):
		self[self.index] = For(name, start, stop, iter_name, direction)
		self.index = self.index + 1

	def code(self, indent_level = 0):
		return DictCode(self, indent_level) + "\n"
