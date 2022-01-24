library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity complete_neurons is

	generic(
		-- internal parallelism
		parallelism		: integer := 16;
		weightsParallelism	: integer := 5;

		-- number of neurons in the layer
		N_neurons		: integer := 400;
		N_addr			: integer := 9;

		-- shift during the exponential decay
		shift			: integer := 1
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
		inh			: in std_logic;
		load_v_th		: in std_logic;
		neuron_addr		: in std_logic_vector(N_addr-1 downto 0);

		-- input
		input_spike		: in std_logic;

		-- input parameters
		v_th_value		: in signed(parallelism-1 downto 0);		
		v_reset			: in signed(parallelism-1 downto 0);		
		inh_weight		: in signed(parallelism-1 downto 0);		
		exc_weights		: in signed(N_neurons*
						weightsParallelism-1 downto 0);

		-- output
		out_spikes		: out std_logic_vector(N_neurons-1 downto 0);
		all_ready		: out std_logic
	);
	
end entity complete_neurons;


architecture behaviour of complete_neurons is

	signal unmasked_select	: std_logic_vector(2**N_addr-1 downto 0);
	signal input_spikes	: std_logic_vector(N_neurons-1 downto 0);
	signal v_th_en		: std_logic_vector(N_neurons-1 downto 0);

	component bare_neurons is

		generic(
			-- internal parallelism
			parallelism		: integer := 16;
			weightsParallelism	: integer := 5;

			-- number of neurons in the layer
			N_neurons		: integer := 400;

			-- shift during the exponential decay
			shift			: integer := 1
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
			v_th_value		: in signed(parallelism-1 downto 0);		
			v_reset			: in signed(parallelism-1 downto 0);		
			inh_weight		: in signed(parallelism-1 downto 0);		
			exc_weights		: in signed(N_neurons*
							  weightsParallelism-1 downto 0);

			-- output
			out_spikes		: out std_logic_vector(N_neurons-1 downto 0);
			all_ready		: out std_logic
		);

	end component bare_neurons;


	
	component decoder is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			encoded_in	: in std_logic_vector(N-1 downto 0);

			-- output
			decoded_out	: out  std_logic_vector(2**N -1 downto 0)
		);

	end component decoder;


	component simple_and_mask is

		generic(
			-- number of input bits on which the mask is applied
			N		: integer := 8		
		);

		port(
			-- input
			input_bits	: in std_logic_vector(N-1 downto 0);
			mask_bit	: in std_logic;

			-- output
			output_bits	: out std_logic_vector(N-1 downto 0)
		);

	end component simple_and_mask;


	component double_and_mask_n is

		generic(
			N		: integer := 3
		);

		port(
			-- input
			input_bits	: in std_logic_vector(N-1 downto 0);
			mask_bit0	: in std_logic;
			mask_bit1	: in std_logic;

			-- output
			output_bits	: out std_logic_vector(N-1 downto 0)
		);

	end component double_and_mask_n;

begin

	neurons_layer	: bare_neurons 

		generic map(
			-- internal parallelism
			parallelism		=> parallelism,
			weightsParallelism	=> weightsParallelism,

			-- number of neurons in the layer
			N_neurons		=> N_neurons,

			-- shift during the exponential decay
			shift			=> shift
		)

		port map(
			-- control input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			stop			=> stop,
			exc_or			=> exc_or,
			exc_stop		=> exc_stop,
			inh_or			=> inh_or,
			inh_stop		=> inh_stop,
			v_th_en			=> v_th_en,

			-- input
			input_spikes		=> input_spikes,

			-- input parameters
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			exc_weights		=> exc_weights,

			-- output
			out_spikes		=> out_spikes,
			all_ready		=> all_ready
		);

	
	neurons_decoder	: decoder 

		generic map(
			N		=> N_addr
		)

		port map(
			-- input
			encoded_in	=> neuron_addr,

			-- output
			decoded_out	=> unmasked_select
		);



	v_th_select	: simple_and_mask 

		generic map(
			-- number of input bits on which the mask is applied
			N		=> N_neurons
		)

		port map(
			-- input
			input_bits	=> unmasked_select(N_neurons-1 downto 0),
			mask_bit	=> load_v_th,
                                                      
			-- output
			output_bits	=> v_th_en
		);


	neurons_select	: double_and_mask_n 

		generic map(
			N		=> N_neurons
		)

		port map(
			-- input
			input_bits	=> unmasked_select(N_neurons-1 downto 0),
			mask_bit0	=> inh,
			mask_bit1	=> input_spike,
                                                      
			-- output
			output_bits	=> input_spikes
		);



end architecture behaviour;
