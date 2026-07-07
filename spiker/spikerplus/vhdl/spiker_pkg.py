from .vhdltools.package_vhdl import Package

class SpikerPackage(Package):

	# Class-wide switch set by VhdlGenerator.generate() before any VHDL
	# object is built. SpikerPackage is instantiated all over the
	# component tree by modules that don't know whether the design is a
	# learning one, and every instance writes the same spiker_pkg.vhd --
	# so all instances of one generation run must agree on it.
	learning_mode = False

	def __init__(self):

		Package.__init__(self, "spiker_pkg")
		self.vhdl()

	def vhdl(self):

		self.pkg_dec.type_list.add("neuron_states", "Enumeration")
		self.pkg_dec.type_list["neuron_states"].add("reset")
		self.pkg_dec.type_list["neuron_states"].add("idle")
		self.pkg_dec.type_list["neuron_states"].add("init")
		self.pkg_dec.type_list["neuron_states"].add("excite")
		self.pkg_dec.type_list["neuron_states"].add("inhibit")
		self.pkg_dec.type_list["neuron_states"].add("fire")
		self.pkg_dec.type_list["neuron_states"].add("leak")

		self.pkg_dec.type_list.add("mi_states", "Enumeration")
		self.pkg_dec.type_list["mi_states"].add("reset")
		self.pkg_dec.type_list["mi_states"].add("idle_wait")
		self.pkg_dec.type_list["mi_states"].add("idle")
		self.pkg_dec.type_list["mi_states"].add("init_wait")
		self.pkg_dec.type_list["mi_states"].add("init")
		self.pkg_dec.type_list["mi_states"].add("sample")
		self.pkg_dec.type_list["mi_states"].add("exc_inh_wait")
		self.pkg_dec.type_list["mi_states"].add("exc_update_full")
		self.pkg_dec.type_list["mi_states"].add("inh_wait_full")
		self.pkg_dec.type_list["mi_states"].add("inh_update_full")
		self.pkg_dec.type_list["mi_states"].add("exc_wait")
		self.pkg_dec.type_list["mi_states"].add("inh_update")
		self.pkg_dec.type_list["mi_states"].add("inh_wait")
		self.pkg_dec.type_list["mi_states"].add("exc_update")
		if SpikerPackage.learning_mode:
			self.pkg_dec.type_list["mi_states"].add("train_wait")
			self.pkg_dec.type_list["mi_states"].add("train_update")

		self.pkg_dec.type_list.add("mc_states", "Enumeration")
		self.pkg_dec.type_list["mc_states"].add("reset")
		self.pkg_dec.type_list["mc_states"].add("idle_wait")
		self.pkg_dec.type_list["mc_states"].add("idle")
		self.pkg_dec.type_list["mc_states"].add("init")
		self.pkg_dec.type_list["mc_states"].add("update_wait")
		self.pkg_dec.type_list["mc_states"].add("network_update")
		if SpikerPackage.learning_mode:
			self.pkg_dec.type_list["mc_states"].add("vote")
