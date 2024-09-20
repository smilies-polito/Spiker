from .text import SingleCodeLine, GenericCodeBlock
from .dict_code import VHDLenum, DictCode
from .format_text import indent
from .variables import VariableList
from .files import FileList
from .if_statement import IfList
from .for_statement import ForList
from .case_statement import CaseList

class SensitivityList(dict):

	def __init__(self, *args):

		self.index = 0
		
		for signal in args:
			self[self.index] = SingleCodeLine(signal, 
					line_end = ", ")
			
			self.index = self.index + 1

	def add(self, signal):
		self[self.index] = SingleCodeLine(signal, 
			line_end = ", ")

		self.index = self.index + 1

	def code(self, indent_level = 0):
		return VHDLenum(self, indent_level)


class Process:

	def __init__(self, name = "", final_wait = False, *args):
		self.name = name
		self.sensitivity_list = SensitivityList(*args)
		self.variables = VariableList()
		self.files = FileList()
		self.bodyHeader = GenericCodeBlock()
		self.if_list = IfList()
		self.for_list = ForList()
		self.case_list = CaseList()
		self.wait_list = None
		self.bodyFooter = GenericCodeBlock()
		self.final_wait = final_wait
	
	def code(self, indent_level = 0):

		hdl_code = ""

		if self.bodyHeader or self.if_list or self.for_list or \
			self.case_list or self.bodyFooter or self.final_wait:

			if(self.name == "" and self.sensitivity_list):
				hdl_code = hdl_code + indent(indent_level) + \
						"process(" + \
						self.sensitivity_list.code() + \
						")\n"

			elif(self.name != "" and self.sensitivity_list):
				hdl_code = hdl_code + indent(indent_level) + \
						self.name + " : process(" + \
						self.sensitivity_list.code() + \
						")\n"

			elif(self.name == "" and not self.sensitivity_list):
				hdl_code = hdl_code + indent(indent_level) + \
						"process\n"

			elif(self.name != "" and not self.sensitivity_list):
				hdl_code = hdl_code + indent(indent_level) + \
						self.name + " : process\n"

			if self.variables:
				hdl_code = hdl_code + self.variables.\
						code(indent_level + 1)

			if self.files:
				hdl_code = hdl_code + self.files.\
						code(indent_level + 1)

			hdl_code = hdl_code + indent(indent_level) + "begin\n\n"

			if self.bodyHeader:
				hdl_code = hdl_code + self.bodyHeader.code(
					indent_level + 1)

			if self.if_list:
				hdl_code = hdl_code + self.if_list.code(
						indent_level + 1)
			if self.for_list:
				hdl_code = hdl_code + self.for_list.code(
						indent_level + 1)
			if self.case_list:
				hdl_code = hdl_code + self.case_list.code(
						indent_level + 1)

			if self.bodyFooter:
				hdl_code = hdl_code + self.bodyFooter.code(
					indent_level + 1)

			if self.final_wait:
				hdl_code = hdl_code + indent(indent_level + 1) \
						+ "wait;\n\n"

			hdl_code = hdl_code + indent(indent_level) + \
					"end process " + self.name + ";\n\n"

		return hdl_code

class ProcessList(dict):

	def add(self, name : str = "", final_wait = False, *args):
		self[name] = Process(name, final_wait, *args)

	def code(self, indent_level : int = 0):
		return DictCode(self, indent_level)
