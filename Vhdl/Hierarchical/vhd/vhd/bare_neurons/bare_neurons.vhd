library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity bare_neurons is

	generic(
		-- internal parallelism
		neuron_bit_width	: integer := 16;
		weights_bit_width	: integer := 5;

		-- number of neurons in the layer
		N_neurons		: integer := 400;

		-- shift during the exponential decay
		shift			: integer := 10
	);

	port(
		-- control input
		clk			: in std_logic;
		rst_n			: in std_logic;		
		start			: in std_logic;		
		stop			: in std_logic;
		exc_or			: in std_logic;
		exc_stop		: in std_logic;
		inh_or			: in std_logic;
		inh_stop		: in std_logic;
		v_th_en			: in std_logic_vector(N_neurons-1 downto 0);

		-- input
		input_spikes		: in std_logic_vector(N_neurons-1 downto 0);

		-- input parameters
		v_th_value		: in signed(neuron_bit_width-1 downto 0);		
		v_reset			: in signed(neuron_bit_width-1 downto 0);		
		inh_weight		: in signed(neuron_bit_width-1 downto 0);		
		exc_weights		: in signed(N_neurons*
						  weights_bit_width-1 downto 0);

		-- output
		out_spikes		: out std_logic_vector(N_neurons-1 downto 0);
		all_ready		: out std_logic
	);

end entity bare_neurons;


architecture behaviour of bare_neurons is

	signal neurons_ready	: std_logic_vector(N_neurons-1 downto 0);

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
			neuron_bit_width	: integer := 16;
			weights_bit_width	: integer := 5;

			-- shift amount
			shift			: integer := 10
		);

		port(
			-- input controls
			clk			: in std_logic;
			rst_n			: in std_logic;
			start			: in std_logic;
			stop			: in std_logic;
			exc_or			: in std_logic;
			exc_stop		: in std_logic;
			inh_or			: in std_logic;
			inh_stop		: in std_logic;
			input_spike		: in std_logic;

			-- to load the threshold
			v_th_en			: in std_logic;

			-- input parameters
			v_th_value		: in signed(neuron_bit_width-1 
							downto 0);
			v_reset			: in signed(neuron_bit_width-1
							downto 0);
			inh_weight		: in signed(neuron_bit_width-1
							downto 0);
			exc_weight		: in signed(weights_bit_width-1
							downto 0);

			-- output
			out_spike		: out std_logic;
			neuron_ready		: out std_logic
		);

	end component neuron;


begin

	

	neurons_ready_and	: generic_and
		generic map(
			N	=> N_neurons
		)

		port map(
			-- input
			and_in	=> neurons_ready,

			-- output
			and_out	=> all_ready
		);


	neurons	: for i in 0 to N_neurons-1
	generate

		neuron_i	: neuron
			generic map(
				-- parallelisms
				neuron_bit_width	=> neuron_bit_width,
				weights_bit_width	=> weights_bit_width,
							       
				-- shift amount    
				shift			=> shift
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
				exc_weight	=> exc_weights((i+1)*
							weights_bit_width-1
							downto
							i*weights_bit_width),
				
				-- to load the threshold
				v_th_en		=> v_th_en(i),
							       
				-- output         
				out_spike	=> out_spikes(i),
				neuron_ready	=> neurons_ready(i)
			);


	end generate neurons;
		



end architecture behaviour;
