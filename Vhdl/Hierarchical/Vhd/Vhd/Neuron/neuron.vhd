library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron is

	generic(
		-- bit-width
		neuron_bit_width	: integer := 16;
		weights_bit_width	: integer := 5;

		-- shift amount
		shift			: integer := 10
	);

	port(

		-- input spike
		exc_spike		: in std_logic;
		inh_spike		: in std_logic;

		-- input controls
		clk			: in std_logic;
		rst_n			: in std_logic;
		restart			: in std_logic;
		load_end		: in std_logic;
		exc			: in std_logic;
		inh			: in std_logic;

		-- direct control on the datapath
		v_th_en			: in std_logic;

		-- input parameters
		v_th_value		: in signed(neuron_bit_width-1 
						downto 0);
		v_reset			: in signed(neuron_bit_width-1
						downto 0);
		inh_weight		: in signed(weights_bit_width-1
						downto 0);
		exc_weight		: in signed(weights_bit_width-1
						downto 0);

		-- output
		load_ready		: out std_logic;
		neuron_ready		: out std_logic;
		out_spike		: out std_logic
	);

end entity neuron;


architecture behaviour of neuron is


	-- from control unit towards datapath
	signal update_sel		: std_logic_vector(1 downto 0);
	signal add_or_sub		: std_logic;
	signal v_update			: std_logic;
	signal v_en			: std_logic;
	signal v_rst_n			: std_logic;

	-- from datapath towards control unit
	signal exceed_v_th		: std_logic;

	-- masking signals
	signal load_en			: std_logic;
	signal masked_v_th_en		: std_logic;
	signal masked_exc_weight	: signed(weights_bit_width-1 downto 0);
	signal masked_inh_weight	: signed(weights_bit_width-1 downto 0);


	component neuron_cu is

		port(
			-- input
			clk		: in std_logic;
			rst_n		: in std_logic;
			restart		: in std_logic;
			load_end	: in std_logic;
			exc		: in std_logic;
			inh		: in std_logic;

			-- input from datapath
			exceed_v_th	: in std_logic;

			-- control output
			update_sel	: out std_logic_vector(1 downto 0);
			add_or_sub	: out std_logic;
			v_update	: out std_logic;
			v_en		: out std_logic;
			v_rst_n		: out std_logic;

			-- output
			load_ready	: out std_logic;
			neuron_ready	: out std_logic;
			out_spike	: out std_logic
		);

	end component neuron_cu;

	component neuron_datapath is

		generic(
			-- bit-width
			neuron_bit_width		: integer := 16;
			weights_bit_width		: integer := 5;

			-- shift amount
			shift				: integer := 1
		);

		port(
			-- input parameters
			v_th_value		: in signed(neuron_bit_width-1 
							downto 0);
			v_reset			: in signed(neuron_bit_width-1 
							downto 0);
			inh_weight		: in signed(weights_bit_width-1
							downto 0);
			exc_weight		: in signed(weights_bit_width-1
							downto 0);

			-- input controls
			clk			: in std_logic;
			update_sel		: in std_logic_vector(1 downto 0);
			add_or_sub		: in std_logic;
			v_update		: in std_logic;
			v_th_en			: in std_logic;
			v_en			: in std_logic;
			v_rst_n			: in std_logic;
			
			-- output
			exceed_v_th		: out std_logic		
		);

	end component neuron_datapath;




begin

	-- allow ro upload v_th only in the load state
	load_ready 	<= load_en;
	masked_v_th_en	<= v_th_en and load_en;


	-- mask the weight using the corresponding spike
	spikes_masking	: process(exc_spike, exc_weight, inh_spike, inh_weight)
	begin

		bitwise_and : for i in 0 to weights_bit_width
		loop
			masked_exc_weight(i)	<= exc_weight(i) and exc_spike;
			masked_inh_weight(i)	<= inh_weight(i) and inh_spike;

		end loop bitwise_and;

	end process spikes_masking;


	control_unit	: neuron_cu
		port map(
			-- input
			clk		=> clk,
			rst_n		=> rst_n,
			restart		=> restart,
			load_end	=> load_end,
			exc		=> exc,
			inh		=> inh,

			-- input from datapath
			exceed_v_th	=> exceed_v_th,

			-- control output
			update_sel	=> update_sel,
			add_or_sub	=> add_or_sub,
			v_update	=> v_update,
			v_en		=> v_en,
			v_rst_n		=> v_rst_n,

			-- output
			load_ready	=> load_en,
			neuron_ready	=> neuron_ready,
			out_spike	=> out_spike
		);


	datapath	: neuron_datapath
		generic map(
			-- bit-width
			neuron_bit_width	=> neuron_bit_width,
			weights_bit_width	=> weights_bit_width,

			-- shift amount
			shift			=> shift
		)
		
		port map(
			-- input parameters
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			exc_weight		=> exc_weight,

			-- input controls
			clk			=> clk,
			update_sel		=> update_sel,
			add_or_sub		=> add_or_sub,
			v_update		=> v_update,
			v_th_en			=> masked_v_th_en,
			v_en			=> v_en,
			v_rst_n			=> v_rst_n,

			-- output
			exceed_v_th		=> exceed_v_th		
		);



end architecture behaviour;
