from .format_text import indent
from .text import GenericCodeBlock
from .dict_code import DictCode, VHDLenum

class Condition:

	"""
	VHDL condition for if/elsif statements.

	Methods:
	--------
	code()	: generate the string of thecondition
	"""

	
	def __init__(self, condition : str, cond_type : str = ""):

		"""
		Parameters:
		-----------
		condition	:str
			Condition for the if/elsif statement

		cond_type	: str, optional
			Type of the condition. Can be empty, "and" or "or"
		"""

		self.condition = condition
		self.cond_type = cond_type

	def code(self) -> str:

		"""
		Generate the condition string.
		"""

		# Normal condition
		if not(self.cond_type):
			hdl_code = self.condition + " "

		# And condition, to combine at least with a normal one
		elif self.cond_type == "and":
			hdl_code = "and " + self.condition + " "

		# Or condition, to combine at least with a normal one
		elif self.cond_type == "or":
			hdl_code = "or " + self.condition + " "


		else:
			print("Types of if condition not supported. Supported "
					"types are \"and\" and \"or\"\n")
			exit(-1)


		return hdl_code



class ConditionsList(dict):

	def __init__(self):
		self.index = 0

	"""
	Dictionary of VHDL conditions.

	Methods:
	--------
	add(condition, cond_type)	: add a condition object to the dictionary
	code()				: generate the string with all the
					conditions
	"""

	def add(self, condition : str, cond_type : str = ""):

		"""
		Add a condition object to the dictionary.

		Parameters:
		-----------
		condition	: str
			Condition for the if/elsif statement
		cond_type	: str
			Type of the condition. Can be empty, "and" or "or"
		"""

		self[self.index] = Condition(condition, cond_type)
		self.index = self.index + 1


	def code(self) -> str:

		"""
		Generate the condition string.
		"""

		return VHDLenum(self)


class If_block:

	"""
	VHDL if block.

	Methods:
	--------
	code(indent_level = 0)	: generate the if statement string
	"""

	def __init__(self):
		self.conditions = ConditionsList()
		self.body = GenericCodeBlock()


	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the if statement string

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""


		hdl_code = ""

		# Generate code only if the condition and body contain something
		if self.conditions and self.body:

			hdl_code = hdl_code + indent(indent_level) + "if " + \
					self.conditions.code() + "\n"
			hdl_code = hdl_code + indent(indent_level) + "then\n\n"
			hdl_code = hdl_code + self.body.code(indent_level + 1)

		return hdl_code


class Elsif_block:

	"""
	VHDL elsif block.

	Methods:
	--------
	code(indent_level = 0)	: generate the elsif statement string
	"""

	def __init__(self):
		self.conditions = ConditionsList()
		self.body = GenericCodeBlock()


	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the elsif statement string

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		hdl_code = ""

		# Generate code only if the condition and body contain something
		if self.conditions and self.body:

			hdl_code = hdl_code + indent(indent_level) + "elsif(" + \
					self.conditions.code() + ")\n"
			hdl_code = hdl_code + indent(indent_level) + "then\n\n"
			hdl_code = hdl_code + self.body.code(indent_level + 1)

		return hdl_code


class Elsif_list(dict):

	"""
	Dictionary of VHDL elsif statements.

	Methods:
	--------
	add()			: add an empty elsif statement to the dictionary
	code(indent_level = 0)	: generate the string to declare all the
				elsif statements
	"""

	def __init__(self):
		self.index = 0

	def add(self):

		"""
		Add an empty elsif statement to the dictionary
		"""

		self[self.index] = Elsif_block()
		self.index = self.index + 1

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string with all the elsif statements

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return DictCode(self, indent_level)


class Else_block:

	"""
	VHDL else block.

	Methods:
	--------
	code(indent_level = 0)	: generate the else statement string
	"""


	def __init__(self):
		self.body = GenericCodeBlock()


	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the else string

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		hdl_code = ""

		# Generate code only if the body contains something
		if self.body:
			hdl_code = hdl_code + indent(indent_level) + "else\n"
			hdl_code = hdl_code + self.body.code(indent_level + 1)

		return hdl_code


class If():

	"""
	VHDL complete if statement.

	Methods:
	--------
	code(indent_level = 0)	: generate the full if statement string
	"""


	def __init__(self):
		self._if_ = If_block()
		self._elsif_ = Elsif_list()
		self._else_ = Else_block()

	def code(self, indent_level = 0):

		"""
		Generate the full if statement string

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""


		hdl_code = ""

		# Generate only if there is an if statement
		if self._if_.code():
			hdl_code = hdl_code + self._if_.code(indent_level)

			# Generate only if there is a list of elsif conditions
			if self._elsif_.code():
				hdl_code = hdl_code + \
					self._elsif_.code(indent_level)

			# Generate only if there is an else condition
			if self._else_.code():
				hdl_code = hdl_code + \
					self._else_.code(indent_level)

			hdl_code = hdl_code + indent(indent_level) + \
					"end if;\n"

		return hdl_code

class IfList(dict):

	def __init__(self):
		self.index = 0

	def add(self):
		self[self.index] = If()
		self.index = self.index + 1

	def code(self, indent_level = 0):
		return DictCode(self, indent_level) + "\n"
