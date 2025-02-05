from .dict_code import DictCode, VHDLenum
from .text import SingleCodeLine

from .format_text import indent

class IncompleteTypeObj:
	def __init__(self, name):
		self.name = name

	def code(self):
		hdl_code = ""
		hdl_code = "type %s;\n" % self.name
		hdl_code = hdl_code + "\n"
		return hdl_code


class EnumerationTypeObj:
	def __init__(self, name, *args):
		self.name = name
		self.type = type
		self.typeElement = dict()
		self.newLine = True
		self.endLine = ",\n"

		if args:
			self.add(args[0])

	def SetNewLine(self, input):
		pass
		if input:
			self.newLine = True
			self.endLine = ",\n"
		else:
			self.newLine = False
			self.endLine = ", "
		for element in self.typeElement:
			self.typeElement[element].line_end = self.endLine

	def add(self, input):
		if isinstance(input, str):
			self.typeElement[input] = SingleCodeLine(input, self.endLine)
		else:
			for element in input:
				self.typeElement[element] = SingleCodeLine(element, self.endLine)

	def code(self, indent_level=1):
		hdl_code = ""
		if self.typeElement:
			hdl_code =  hdl_code + indent(indent_level) + "type %s is ( " % self.name
			hdl_code = hdl_code + "\n"
			hdl_code = hdl_code + "%s" % VHDLenum(self.typeElement, indent_level+1)
			hdl_code = hdl_code + "\n"
			hdl_code = hdl_code + indent(indent_level) + ");\n\n"
		return hdl_code


class ArrayTypeObj:
	def __init__(self, name, *args):
		self.name = name
		self.arrayRange = args[0]
		self.arrayType = args[1]

	def code(self, indent_level = 1):
		hdl_code = ""
		hdl_code = indent(indent_level) + "type %s is array (%s) of "\
		"%s;\n" % (self.name, self.arrayRange, self.arrayType)
		hdl_code = hdl_code + "\n"
		return hdl_code


class RecordTypeObj:
	def __init__(self, name, *args):
		self.name = name

		if args:
			if isinstance(args[0], GenericList):
				self.element = args[0]
		else:
			self.element = GenericList()

	def add(self, name, type, init=None):
		self.element.add(name, type, init)

	def code(self):
		hdl_code = GenericCodeBlock()
		hdl_code.add("type %s is record" % self.name)
		for j in self.element:
			hdl_code.add(indent(1) + "%s : %s;" % (self.element[j].name, self.element[j].type) )
		hdl_code.add( "end record %s;" % self.name )
		return hdl_code.code()

class SubTypeObj:
	def __init__(self, name, *args):
		self.name = name
		self.ofType = args[0]
		self.element = GenericList()

	def add(self, name, type, init=None):
		self.element.add(name, type, init)

	def code(self):
		hdl_code = ""
		hdl_code = hdl_code + indent(1) + "subtype %s is %s (\n" % (self.name, self.ofType)
		i = 0
		for j in self.element:
			i += 1
			if (i == len(self.element)):
				hdl_code = hdl_code + indent(2) + "%s (%s)); \n" % (self.element[j].name, self.element[j].type)
			else:
				hdl_code = hdl_code + indent(2) + "%s (%s),\n" % (self.element[j].name, self.element[j].type)
				hdl_code = hdl_code + "\n"
		return hdl_code


class CustomTypeList(dict):
	def add(self, name, c_type, *args):
		if "Array" in c_type:
			self[name] = ArrayTypeObj(name, *args)

		elif "Enumeration" in c_type:
			self[name] = EnumerationTypeObj(name, *args)
		elif "Record" in c_type:
			self[name] = RecordTypeObj(name, *args)
		elif "SubType" in c_type:
			self[name] = SubTypeObj(name, *args)
		else:
			self[name] = IncompleteTypeObj(name)

	def code(self, indent_level=0):
		return DictCode(self, indent_level = indent_level)
