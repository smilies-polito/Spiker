from .format_text import indent
from .dict_code import VHDLenum, DictCode
from .text import SingleCodeLine

from .map_signals import MapList

class Instance():

	def __init__(self, component, instance_name):
		self.instance_name = instance_name
		self.name = component.entity.name
		self.generic_list = component.entity.generic
		self.port_list = component.entity.port
		self.g_map = {}
		self.p_map = {}

	
	def generic_map(self, mode = "auto", *args, **kwargs):
		self.g_map = MapList(self.generic_list, mode, *args, **kwargs)

	def port_map(self, mode = "auto", *args, **kwargs):
		self.p_map = MapList(self.port_list, mode, *args, **kwargs)

	def code(self, indent_level = 0):

		hdl_code = ""

		if self.p_map:

			hdl_code = indent(indent_level) + self.instance_name + \
					" : " + self.name + "\n"

			if self.g_map:
				hdl_code = hdl_code + \
					indent(indent_level + 1) + \
					"generic map(\n"

				hdl_code = hdl_code + \
					self.g_map.code(indent_level + 2)

				hdl_code = hdl_code + indent(indent_level + 1) \
						+ ")\n"

			hdl_code = hdl_code + indent(indent_level + 1) + \
					"port map(\n"


			hdl_code = hdl_code + self.p_map.code(indent_level + 2)

			hdl_code = hdl_code + indent(indent_level + 1) + \
					");\n\n"


		return hdl_code

class InstanceList(dict):

	def add(self, component, instance_name : str):
		self[instance_name] = Instance(component, instance_name)

	def code(self, indent_level : int = 0):
		return DictCode(self, indent_level)
