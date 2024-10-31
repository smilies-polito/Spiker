from .format_text import indent
from .dict_code import DictCode

class FileObj:

	def __init__(self, name : str = "out_file", file_type : str = "text",
			mode = "write_mode", filename = "\"output.txt\""):

		self.name = name
		self.file_type = file_type
		self.mode = mode
		self.filename = filename


	def code(self, indent_level : int = 0):

		return indent(indent_level) + ("file %s : %s open %s is %s;\n" \
				% (self.name, self.file_type, self.mode,
				self.filename))

class FileList(dict):


	def add(self, name = "out_file", file_type = "text", mode =
			"write_mode", filename = "\"output.txt\""):

		self[name] = FileObj(
			name 		= name, 
			file_type	= file_type,
			mode		= mode,
			filename 	= filename
		)

	def code(self, indent_level : int = 0) -> str:

		return DictCode(self, indent_level)
