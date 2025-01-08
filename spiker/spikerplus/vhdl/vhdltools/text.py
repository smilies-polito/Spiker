from .format_text import indent
from .dict_code import DictCode

from typing import TypeVar, Generic, Union

VhdlClass = TypeVar("Whichever class"
		"imlementing the code() method")

class SingleCodeLine:

	"""
	Generate a single line of text.

	Methods:
	--------
	code(indent_level = 0)	: return the generated string of text

	"""


	def __init__(self, value : str, line_end : str = ""):

		"""
		Parameters:
		-----------
		value		: str
			Text content of the output string
		line_end	: str, optional
			Terminating character(s) of the output string. By
			default it is empty
		"""

		self.value = value
		self.line_end = line_end

	def code(self, indent_level : int = 0) -> str:

		"""
		Compose the output string.

		Parameters:
		-----------
		indent_level	: int, optional
			How many indent level to put before the string
		"""

		return indent(indent_level) + self.value + self.line_end



class GenericCodeBlock(dict):

	"""
	Generate a block of text. The block is treated as a dictionary with
	integer keys.

	Methods:
	--------
	add(text, line_end = "\n")	: add a single line of text to the block
	code(indent_level = 0)		: return the generated string of text
	"""

	def __init__(self):

		"""
		Begin the insertion from the element 0.
		"""

		self.index = 0

	def add(self, text : Union[str, VhdlClass], line_end : str = "\n"):

		"""
		Add a single line of text to the block.

		Parameters:
		-----------
		text	: str
			Text content of the line to add.
		line_end	: str, optional
			Terminating character(s) of the output string. By
			default it is newline

		"""

		if type(text) == str:
			self[self.index] = SingleCodeLine(text, line_end)

		elif hasattr(text, "code") and callable(text.code):
			self[self.index] = text

		else:
			print("Invalid text argument to GenericCodeBlock\n")
			exit(-1)

		self.index = self.index + 1

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate block of text.

		Parameters:
		-----------
		indent_level	: int, optional
			How many indent level to put before the string
		"""

		return DictCode(self, indent_level) + "\n"
