from .custom_types import CustomTypeList
from .write_file import write_file

class PackageDeclaration():

	def __init__(self, name):
		self.name = name
		self.type_list = CustomTypeList()

	def code(self, indent_level = 0):

		hdl_code = ""

		hdl_code += "package " + self.name + " is\n\n"

		if self.type_list:
			hdl_code += self.type_list.code(indent_level + 1)

		hdl_code += "end package " + self.name + ";\n"

		return hdl_code

class Package():

	def __init__(self, name):
		self.name = name
		self.pkg_dec = PackageDeclaration(name)

	def code(self, indent_level = 0):

		hdl_code = ""

		hdl_code += self.pkg_dec.code(indent_level)

		return hdl_code

	def write_file(self, output_dir = "output", rm = False):
		write_file(self, output_dir = output_dir, rm = rm)
