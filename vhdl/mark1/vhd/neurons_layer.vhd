library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neurons_layer is

	generic(
		-- neurons counter parallelism
		N_cnt		: integer := 2;

		-- internal parallelism
		N		: integer := 8;

		-- number of neurons in the layer
		layer_size	: integer := 3
	);

	port(
		-- control input
		clk		: in std_logic;
		rst_n		: in std_logic;		
		start		: in std_logic;		
		start1		: in std_logic;		
		start2		: in std_logic;		
		rest_en		: in std_logic;
		neuron_cnt	: in std_logic_vector(N_cnt-1 downto 0);

		-- input parameters
		v_th_0		: in signed(N-1 downto 0);		
		v_reset		: in signed(N-1 downto 0);		
		inh_weight	: in signed(N-1 downto 0);		
		v_th_plus	: in signed(N-1 downto 0);		
		exc_weights	: in signed(layer_size*N-1 downto 0);

		-- output
		out_spikes	: out std_logic_vector(layer_size-1 downto 0);
		layer_ready	: out std_logic
	);

end entity neurons_layer;


architecture behaviour of neurons_layer is

	signal mask_neuron	: std_logic_vector(2**N_cnt-1 downto 0);
	signal neuron_ready	: std_logic_vector(layer_size-1 downto 0);

	component decoder is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			encoded_in	: in std_logic_vector(N-1 downto 0);

			-- output
			decoded_out	: out  std_logic_vector(0 to 2**N -1)
		);

	end component decoder;



	component neuron is

		generic(
			-- parallelism
			N		: integer := 8;

			-- shift amount
			shift		: integer := 1
		);

		port(
			-- input controls
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			start1		: in std_logic;
			start2		: in std_logic;
			rest_en		: in std_logic;
			mask_neuron	: in std_logic;
			input_spike	: in std_logic;

			-- input parameters
			v_th_0		: in signed(N-1 downto 0);
			v_reset		: in signed(N-1 downto 0);
			inh_weight	: in signed(N-1 downto 0);
			exc_weight	: in signed(N-1 downto 0);
			v_th_plus	: in signed(N-1 downto 0);

			-- output
			out_spike	: out std_logic;
			neuron_ready	: out std_logic
		);

	end component neuron;


begin



end architecture behaviour;
