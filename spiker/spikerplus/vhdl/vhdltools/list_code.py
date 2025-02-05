from .format_text import indent

def VHDLenum_list(objects_list : list, indent_level : int = 0) -> str:

	"""
	Generate a string concatenating all the elements of a list of
	objects, removing the last terminating character (; or ,).

	Parameters:
	-----------
	objects_list	: list of objects.
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

	for element in objects_list:
		i = i+1
		if (i == len(objects_list)):

			# Remove terminating character
			tmp = element.code()
			tmp = tmp.replace(";", "")
			tmp = tmp.replace(",", "")

			hdl_code = hdl_code + indent(indent_level) + tmp

		else:
			hdl_code = hdl_code + indent(indent_level) + \
					element.code()
	return hdl_code


def ListCode(objects_list : list, indent_level : int = 0) -> str:

	"""
	Generate a string concatenating all the elements of a list of
	objects.

	Parameters:
	-----------
	objects_list	: list of objects.
		The object must implement the code() method.
	indent_level	: int, optional
		Number of indentations to insert between printed elements.

	Return:
	-------
	hdl_code	: str
		Generated string
	"""

	hdl_code = ""

	for element in objects_list:
		hdl_code = hdl_code + indent(indent_level) + \
				element.code()
	
	return hdl_code
