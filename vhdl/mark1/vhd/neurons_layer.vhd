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
		layer_size	: integer := 3;

		-- shift during the exponential decay
		shift		: integer := 1
	);

	port(
		-- control input
		clk		: in std_logic;
		rst_n		: in std_logic;		
		start		: in std_logic;		
		start1		: in std_logic;		
		start2		: in std_logic;		
		rest_en		: in std_logic;
		input_spike	: in std_logic;
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


	neuron_decoder	: decoder
		generic map(
			N	=> N_cnt		
		)
		port map(
			-- input
			encoded_in	=> neuron_cnt,

			-- output
			decoded_out	=> mask_neuron
		);

	layer_ready_and	: generic_and
		generic map(
			N	=> layer_size
		)

		port map(
			-- input=>
			and_in	=> neuron_ready,

			-- outpu=>
			and_out	=> layer_ready
		);


	neurons	: for i in 0 to layer_size-1
	generate

		neuron_i	: neuron
			generic map(
			-- parallelism
			N		=> N,
                                                       
			-- shift amount    
			shift		=> shift
			)                                      
							       
			port map(
				-- input control
				clk		=> clk,
				rst_n		=> rst_n,
				start		=> start,
				start1		=> start1,
				start2		=> start2,
				rest_en		=> rest_en,
				mask_neuron	=> mask_neuron(i),
				input_spike	=> input_spike,
							       
				-- input parameters
				v_th_0		=> v_th_0,
				v_reset		=> v_reset,
				inh_weight	=> inh_weight,
				exc_weight	=> exc_weights((i+1)*N-1 downto
							i*N),
				v_th_plus	=> v_th_plus,
							       
				-- output         
				out_spike	=> out_spikes(i),
				neuron_ready	=> neuron_ready(i)
			);


	end generate neurons;
		



end architecture behaviour;
