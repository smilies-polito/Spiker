import brian2 as b2

parametersDict = {

	# Excitatory rest potential
	"vRest_exc"		: -65.0,

	# Inhibitory rest potential
	"vRest_inh"		: -60.0,

	# Excitatory refractory period
	"tRefrac_exc"		: 5.*b2.ms,

	# Inhibitory refractory period
	"tRefrac_inh"		: 2*b2.ms
}

weightInitDict = {
	"exc2inh"	: 22.5,
	"inh2exc"	: -120
}
