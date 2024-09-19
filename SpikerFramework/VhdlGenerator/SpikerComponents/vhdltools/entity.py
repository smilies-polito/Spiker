from .format_text import indent
from .generic import GenericList
from .port import PortList


class Entity:

	def __init__(self, name):
		self.name = name
		self.generic = GenericList()
		self.port = PortList()

	def code(self, indent_level = 0):

		hdl_code = indent(indent_level) + ("entity %s is\n" % self.name)

		if (self.generic):
			hdl_code = hdl_code + indent(indent_level+1) + \
					("generic (\n")
			hdl_code = hdl_code + self.generic.code(indent_level+2)
			hdl_code = hdl_code + indent(indent_level+1) + (");\n")


		if (self.port):
			hdl_code = hdl_code + indent(indent_level+1) + \
					("port (\n")
			hdl_code = hdl_code + self.port.code(indent_level+2)
			hdl_code = hdl_code + indent(indent_level+1) + (");\n")

		hdl_code = hdl_code + indent(indent_level) + \
				("end entity %s;\n" % self.name)
		hdl_code = hdl_code + "\n"

		return hdl_code
