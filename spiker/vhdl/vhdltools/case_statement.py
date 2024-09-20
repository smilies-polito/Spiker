from .format_text import indent
from .text import SingleCodeLine
from .text import GenericCodeBlock
from .dict_code import DictCode, VHDLenum



class When:

	def __init__(self, choice = "others"):
		self.choice = SingleCodeLine(choice)
		self.body = GenericCodeBlock()


	def code(self, indent_level : int = 0) -> str:

		hdl_code = ""

		# Generate code only if the condition and body contain something
		if self.choice and self.body:

			hdl_code = hdl_code + indent(indent_level) + "when " + \
					self.choice.code() + " =>\n"
			hdl_code = hdl_code + self.body.code(indent_level + 1) + "\n"

		return hdl_code

class WhenList(dict):

	def add(self, choice = "others"):
		self[choice] = When(choice)

	def code(self, indent_level = 0) -> str:

		"""
		Generate the condition string.
		"""

		return DictCode(self, indent_level)

class Case():

	def __init__(self, expression):
		self.expression = expression
		self.when_list = WhenList()
		self.others = When("others")

	def code(self, indent_level = 0):

		hdl_code = ""

		if self.expression and self.others.body:

			hdl_code = hdl_code + indent(indent_level) + "case " + \
				self.expression + " is\n\n"

			hdl_code = hdl_code + self.when_list.code(indent_level +
					1)

			hdl_code = hdl_code + self.others.code(indent_level + 1)

			hdl_code = hdl_code + indent(indent_level) +\
					"end case;\n\n"


		return hdl_code


class CaseList(dict):

	def add(self, expression):
		self[expression] = Case(expression)


	def code(self, indent_level = 0) -> str:
		return DictCode(self, indent_level)
