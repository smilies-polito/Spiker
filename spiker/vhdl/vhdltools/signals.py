from .format_text import indent
from .dict_code import DictCode

class SignalObj:

	"""
	Declare a VHDL signal.

	Methods:
	--------
	code(indent_level = 0)	: generate the string to declare the signal
	"""

	def __init__(self, name : str, signal_type : str, *args : str):

		"""
		Parameters:
		-----------
		name		: str
			Name of the signal
		signal_type	: str
			Type of the signal
		*args		: str
			List of default values connected to the signal. In
			practice only the first is used.
		"""

		self.name = name
		self.signal_type = signal_type

		if args:
			self.value = args[0]
		else:
			self.value = ""

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare the signal

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		# Default value
		if self.value:
			return indent(indent_level) + ("signal %s : %s := \
					%s;\n" % (self.name, self.signal_type,
						self.value))
		# No default value
		else:
			return indent(indent_level) + \
					("signal %s : %s;\n" % \
					(self.name, self.signal_type))


class SignalList(dict):

	"""
	Dictionary of VHDL signals.

	Methods:
	--------
	add(name, signal_type, *args)	: add a signal object to the 
					dictionary
	code(indent_level = 0)		: generate the string to declare all the
					signals
	"""

	def add(self, name, signal_type, *args):

		"""
		Add a signal object to the dictionary.

		Parameters:
		-----------
		name		: str
			Name of the signal
		signal_type	: str
			Type of the signal
		*args		: str
			List of default values connected to the signal. In
			practice only the first is used.

		"""

		self[name] = SignalObj(name, signal_type, *args)

	def code(self, indent_level : int = 0) -> str:

		"""
		Generate the string to declare all the signals

		Parameters:
		----------
		indent_level	: int
			Level of indentation to insert before the string
		"""

		return DictCode(self, indent_level)
