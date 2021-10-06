import brian2 as b2

parametersDict = {

	# Excitatory rest potential
	"vRest_exc"		: -65.0,	# mV

	# Inhibitory rest potential
	"vRest_inh"		: -60.0,	# mV

	# Excitatory refractory period
	"tRefrac_exc"		: 0*b2.ms,

	# Inhibitory refractory period
	"tRefrac_inh"		: 0*b2.ms, 
}

weightInitDict = {
	"exc2inh"	: 22.5,			# mV
	"inh2exc"	: -15			# mV
}
