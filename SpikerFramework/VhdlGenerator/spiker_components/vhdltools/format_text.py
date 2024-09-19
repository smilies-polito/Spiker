def indent(value = 0, tabsize = 4):

	txt = ""

	if value > 0:
		j = 0
		while j < tabsize * value:
			txt = txt + " "
			j += 1
	return txt
