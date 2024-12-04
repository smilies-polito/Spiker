from .format_text import indent

def VHDLenum(objects_dict : dict, indent_level : int = 0) -> str:

	"""
	Generate a string concatenating all the elements of a dictionary of
	objects, removing the last terminating character (; or ,).

	Parameters:
	-----------
	objects_dict	: dictionary of objects.
		The object must implement the code() method.
	indent_level	: int, optional
		Number of indentations to insert between printed elements.

	Return:
	-------
	hdl_code	: str
		Generated string
	"""

	hdl_code = ""

	i = 0

	for key in objects_dict:
		i = i+1
		if (i == len(objects_dict)):

			# Remove terminating character
			tmp = objects_dict[key].code()
			tmp = tmp.replace(";", "")
			tmp = tmp.replace(",", "")

			hdl_code = hdl_code + indent(indent_level) + tmp

		else:
			hdl_code = hdl_code + indent(indent_level) + \
					objects_dict[key].code()
	return hdl_code




def DictCode(objects_dict : dict, indent_level : int = 0) -> str:

	"""
	Generate a string concatenating all the elements of a dictionary of
	objects.

	Parameters:
	-----------
	objects_dict	: dictionary of objects.
		The object must implement the code() method.
	indent_level	: int, optional
		Number of indentations to insert between printed elements.

	Return:
	-------
	hdl_code	: str
		Generated string
	"""

	hdl_code = ""

	for key in objects_dict:
		hdl_code = hdl_code + objects_dict[key].code(indent_level)
	
	return hdl_code
