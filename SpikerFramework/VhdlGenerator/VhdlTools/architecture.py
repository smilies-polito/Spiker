from .constant import ConstantList
from .component import ComponentList
from .text import GenericCodeBlock
from .process import ProcessList
from .instance import InstanceList
from .signals import SignalList
from .custom_types import CustomTypeList
from .format_text import indent

class Architecture:

	def __init__(self, name, entity_name):
		self.name = name
		self.entityName = entity_name
		self.customTypes = CustomTypeList()
		self.signal = SignalList()
		self.constant = ConstantList()
		self.component = ComponentList()
		self.declarationHeader = GenericCodeBlock()
		self.declarationFooter = GenericCodeBlock()
		self.bodyCodeHeader = GenericCodeBlock()
		self.processes = ProcessList()
		self.instances = InstanceList()
		self.bodyCodeFooter = GenericCodeBlock()

	def code(self, indent_level = 0):

		hdl_code = ""
		hdl_code = indent(indent_level) + ("architecture %s of %s is\n" 
				% (self.name, self.entityName))
		hdl_code = hdl_code + "\n"
		hdl_code = hdl_code + "\n"

		if (self.declarationHeader):
			hdl_code = hdl_code + \
				self.declarationHeader.code(indent_level + 1)
			hdl_code = hdl_code + "\n"

		if (self.customTypes):
			hdl_code = hdl_code + self.customTypes.code(
					indent_level + 1)
			hdl_code = hdl_code + "\n"

		if (self.constant):
			hdl_code = hdl_code + self.constant.code(indent_level +
					1)
			hdl_code = hdl_code + "\n"

		if (self.component):
			hdl_code = hdl_code + self.component.code(indent_level +
					1)
			hdl_code = hdl_code + "\n"

		if (self.signal):
			hdl_code = hdl_code + self.signal.code(indent_level + 1)
			hdl_code = hdl_code + "\n"


		if (self.declarationFooter):
			hdl_code = hdl_code + self.declarationFooter.code(
					indent_level + 1)
			hdl_code = hdl_code + "\n"
			hdl_code = hdl_code + "\n"
			hdl_code = hdl_code + "\n"

		hdl_code = hdl_code + indent(indent_level) + ("begin\n\n")


		if (self.bodyCodeHeader):
			hdl_code = hdl_code + self.bodyCodeHeader.code(
					indent_level + 1)
			hdl_code = hdl_code + "\n"
			
		if self.processes:

			hdl_code = hdl_code + self.processes.code(indent_level +
					1)
			hdl_code = hdl_code + "\n"

		if (self.instances):
			hdl_code = hdl_code + self.instances.code(
					indent_level + 1)
			hdl_code = hdl_code + "\n"

		if (self.bodyCodeFooter):
			hdl_code = hdl_code + self.bodyCodeFooter.code(
					indent_level + 1)
			hdl_code = hdl_code + "\n"

		hdl_code = hdl_code + indent(indent_level) + \
				("end architecture %s;\n" % self.name)
		hdl_code = hdl_code + "\n"

		return hdl_code
