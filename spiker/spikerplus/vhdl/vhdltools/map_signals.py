from .format_text import indent
from .dict_code import VHDLenum

class MapObj:

	def __init__(self, port, signal, conn_range = ""):
		self.port = port
		self.signal = signal
		self.conn_range = conn_range

	def code(self, indent_level = 0):
		return self.port + self.conn_range + " => " + \
			self.signal + ",\n"

class MapList(dict):

	def __init__(self, elements_list, mode = "auto", *args, **kwargs):

		self.elements_list = elements_list

		if mode == "auto":
		
			for key in self.elements_list:

				name = self.elements_list[key].name

				self[name] = MapObj(name, name)

		elif mode == "pos":

			if len(args) != len(self.elements_list):
				print("Error, wrong number of elements in"
						" signals mapping\n")
				exit(-1)

			else:

				for source, target in zip(args, self.elements_list):

					if type(source) == str:
						source_name = source

					elif hasattr(source, "name"):
						source_name = source.name

					else:
						raise ValueError("Wrong source in"
							" signal mapping")

					target_name = self.elements_list[target].name

					self[target_name] = MapObj(target_name,
							source_name)

		elif mode == "key":

			if len(kwargs) != len(self.elements_list):

				mapped = []
				for key in kwargs:
					mapped.append(key)

				print("Elements to map: " +
					str(self.elements_list) + "\nElements"
					" mapped: " + str(mapped) + "\n")
				raise ValueError("Wrong number of elements in"
						" signals mapping")

			
			for target in self.elements_list:

				if self.elements_list[target].name in kwargs:

					if type(kwargs[target]) == str:
						source_name = kwargs[target]

					elif hasattr(kwargs[target], "name"):
						source_name = \
							kwargs[target].name

					else:
						raise ValueError("Wrong source in"
							" signal mapping")

					target_name = self.elements_list[target].name

					self[target_name] = MapObj(target_name,
							source_name)

				else:
					raise ValueError("Signal not mapped")

		elif mode == "self":

			for key in self.elements_list:

				name	= self.elements_list[key].name
				value	= str(self.elements_list[key].value)

				self[name] = MapObj(name, value)


		elif mode == "no":
			pass

		else:
			raise ValueError("Wrong instance mode")


	def add(self, target_name, source, conn_range = ""):

		present = False

		for target in self.elements_list:
			if target_name == self.elements_list[target].name:
				present = True

		if present:

			if conn_range and target_name in self:
				print(self.code())
				del self[target_name]

			if type(source) == str:
				self[target_name + conn_range] = \
					MapObj(target_name, source, conn_range)

			elif hasattr(source, "name"):
				self[target_name + conn_range] = \
					MapObj(target_name, source.name,
					conn_range)

			else:
				raise ValueError("Wrong source in signal mapping")


		else:
			raise ValueError("Signal " + target_name + " not present")



	def code(self, indent_level = 0):
		return VHDLenum(self, indent_level)

