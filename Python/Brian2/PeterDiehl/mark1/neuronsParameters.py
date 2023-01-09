import brian2 as b2

parametersDict = {

	# Excitatory rest potential
	"vRest_exc"	: -65.*b2.mV,

	# Inhibitory rest potential
	"vRest_inh"	: -60.*b2.mV,


	# Excitatory refractory period
	"tRefrac_exc"	: 5.*b2.ms,

	# Inhibitory refractory period
	"tRefrac_inh"	: 2.*b2.ms

}

weightInitDict = {
	"exc2inh"	: 10.4,
	"inh2exc"	: 17.4
}
