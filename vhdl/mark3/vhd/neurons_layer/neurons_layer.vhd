library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neurons_layer is

	generic(
		-- neurons counter parallelism
		N_cnt		: integer := 10;

		-- internal parallelism
		N		: integer := 16;
		N_weights	: integer := 5;

		-- number of neurons in the layer
		layer_size	: integer := 400;

		-- shift during the exponential decay
		shift		: integer := 1
	);

	port(
		-- control input
		clk		: in std_logic;
		rst_n		: in std_logic;		
		start		: in std_logic;		
		stop		: in std_logic;
		exc_or		: in std_logic;
		exc_stop	: in std_logic;
		inh_or		: in std_logic;
		inh_stop	: in std_logic;
		input_spikes	: in std_logic_vector(layer_size-1 downto 0);

		-- input parameters
		v_th_value	: in signed(N-1 downto 0);		
		v_reset		: in signed(N-1 downto 0);		
		inh_weight	: in signed(N-1 downto 0);		
		exc_weights	: in signed(layer_size*N_weights-1 downto 0);

		-- output
		out_spikes	: out std_logic_vector(layer_size-1 downto 0);
		neurons_ready	: out std_logic
	);

end entity neurons_layer;


architecture behaviour of neurons_layer is

	signal neuron_ready	: std_logic_vector(layer_size-1 downto 0);

	component generic_and is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			and_in	: in std_logic_vector(N-1 downto 0);

			-- output
			and_out	: out std_logic
		);

	end component generic_and;




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
			stop		: in std_logic;
			exc_or		: in std_logic;
			exc_stop	: in std_logic;
			inh_or		: in std_logic;
			inh_stop	: in std_logic;
			input_spike	: in std_logic;

			-- input parameters
			v_th_value	: in signed(N-1 downto 0);
			v_reset		: in signed(N-1 downto 0);
			inh_weight	: in signed(N-1 downto 0);
			exc_weight	: in signed(N-1 downto 0);

			-- output
			out_spike	: out std_logic;
			neuron_ready	: out std_logic
		);

	end component neuron;


begin

	

	neurons_ready_and	: generic_and
		generic map(
			N	=> layer_size
		)

		port map(
			-- input
			and_in	=> neuron_ready,

			-- output
			and_out	=> neurons_ready
		);


	neurons	: for i in 0 to layer_size-1
	generate

		neuron_i	: neuron
			generic map(
			-- parallelism
			N		=> N,
			N_weights	=> N_weights,
                                                       
			-- shift amount    
			shift		=> shift
			)                                      
							       
			port map(
				-- input control
				clk		=> clk,
				rst_n		=> rst_n,
				start		=> start,
				stop		=> stop,
				exc_or	       	=> exc_or,
				exc_stop       	=> exc_stop,
				inh_or	        => inh_or,
				inh_stop        => inh_stop,
				input_spike	=> input_spikes(i),
							       
				-- input parameters
				v_th_value	=> v_th_value,
				v_reset		=> v_reset,
				inh_weight	=> inh_weight,
				exc_weight	=> exc_weights((i+1)*N_weights-1 downto
							i*N),
							       
				-- output         
				out_spike	=> out_spikes(i),
				neuron_ready	=> neuron_ready(i)
			);


	end generate neurons;
		



end architecture behaviour;
