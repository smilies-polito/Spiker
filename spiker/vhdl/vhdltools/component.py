from .generic import GenericList
from .port import PortList
from .format_text import indent
from .dict_code import DictCode

class ComponentObj:

	"""
	VHDL component definition.

	Methods:
	--------
	code(indent_level = 0)	: generate the string to declare the component

	"""


	def __init__(self, vhdl_block):

		"""
		Parameters:
		-----------
		name		: str
			Name of the component
		"""

		self.name = vhdl_block.entity.name
		self.generic = vhdl_block.entity.generic
		self.port = vhdl_block.entity.port


	def code(self, indent_level : int = 0):

		"""
		Generate the string to declare the component

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""


		hdl_code = ""
		hdl_code = hdl_code + indent(indent_level) + \
				("component %s is\n" % self.name)

		if (self.generic):
			hdl_code = hdl_code + \
					indent(indent_level+1) + \
					("generic (\n")
			hdl_code = hdl_code + \
					self.generic.code(indent_level + 2)
			hdl_code = hdl_code + \
					indent(indent_level+1) + (");\n")
		if (self.port):
			hdl_code = hdl_code + \
					indent(indent_level+1) + ("port (\n")
			hdl_code = hdl_code + \
					self.port.code(indent_level+2)
			hdl_code = hdl_code + \
					indent(indent_level+1) + (");\n")

		hdl_code = hdl_code + \
				indent(indent_level) + \
				("end component;\n")
		hdl_code = hdl_code + "\n"

		return hdl_code



class ComponentList(dict):

	def add(self, vhdl_block):
		self[vhdl_block.entity.name] = ComponentObj(vhdl_block)

	def code(self, indent_level : int = 0):

		"""
		Generate the string to declare all the components

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return DictCode(self, indent_level)
