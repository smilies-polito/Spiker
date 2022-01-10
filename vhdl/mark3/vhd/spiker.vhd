library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spiker is

	port(
					
		-- input
		clk			: in std_logic;
		rst_n			: in std_logic;	
		start			: in std_logic;	


		-- input parameters
		v_th_value		: in signed(15 downto 0);		

		-- output
		sample			: out std_logic;
		ready			: out std_logic
	);

end entity spiker;

architecture behaviour of spiker is

	constant neuronParallelism	: integer := 16;
	constant weightParallelism	: integer := 5;
	constant N_inputs		: integer := 784;
	constant N_exc_cnt		: integer := 11;
	constant N_neurons		: integer := 400;
	constant N_inh_cnt		: integer := 10;
	constant N_cycles		: integer := 3500;
	constant N_cycles_cnt		: integer := 12;
	constant shift			: integer := 1;

	constant v_reset		: integer := 5*2**10;
	constant inh_weight		: integer := -15;

	signal input_spikes		: std_logic_vector(N_inputs-1 downto 0);
	signal exc_weights		: signed(N_neurons*5-1 downto 0);
	signal exc_cnt			: std_logic_vector(N_exc_cnt-1 
						downto 0);
	signal outSpikes		: std_logic_vector(N_neurons-1 
						downto 0);



	signal v_reset_sig		: signed(neuronParallelism-1 downto 0);	
	signal inh_weight_sig		: signed(neuronParallelism-1 downto 0);	
	signal N_inputs_sig		: std_logic_vector(N_exc_cnt-1 downto 0);	
	signal N_neurons_sig		: std_logic_vector(N_inh_cnt-1 downto 0);	
	signal N_cycles_sig		: std_logic_vector(N_cycles_cnt-1 downto 0);	




	component layer is

		generic(

			-- internal parallelism
			parallelism		: integer := 16;
			weightParallelism	: integer := 5;

			-- excitatory spikes
			input_parallelism	: integer := 784;
			N_exc_cnt		: integer := 11;

			-- inhibitory spikes
			layer_size		: integer := 400;
			N_inh_cnt		: integer := 10;

			-- elaboration steps
			N_cycles_cnt		: integer := 12;

			-- exponential decay shift
			shift			: integer := 1
				
		);

		port(
			-- input
			clk			: in std_logic;
			rst_n			: in std_logic;	
			start			: in std_logic;	
			input_spikes		: in std_logic_vector
							(input_parallelism-1 downto 0);

			-- input parameters
			v_th_value		: in signed(parallelism-1 downto 0);		
			v_reset			: in signed(parallelism-1 downto 0);	
			inh_weight		: in signed(parallelism-1 downto 0);		
			exc_weights		: in signed
						(layer_size*weightParallelism-1 downto 0);

			-- number of inputs, neurons and cycles
			N_inputs		: in std_logic_vector
							(N_exc_cnt-1 downto 0);
			N_neurons		: in std_logic_vector
							(N_inh_cnt-1 downto 0);
			N_cycles		: in std_logic_vector
							(N_cycles_cnt-1 downto 0);

			-- output
			exc_cnt			: out std_logic_vector
							(N_exc_cnt-1 downto 0);
			out_spikes		: out std_logic_vector
							(layer_size-1 downto 0);
			sample			: out std_logic;
			ready			: out std_logic
		);

	end component layer;


begin

	v_reset_sig	<= to_signed(v_reset, neuronParallelism);
	inh_weight_sig	<= to_signed(inh_weight, neuronParallelism);


	N_inputs_sig	<= std_logic_vector(to_signed(N_inputs,
				N_exc_cnt));
	N_neurons_sig	<= std_logic_vector(to_signed(N_neurons,
				N_inh_cnt));
	N_cycles_sig	<= std_logic_vector(to_signed(N_cycles,
				N_cycles_cnt));


	layer_1		: layer 

			generic map(

				-- internal parallelism
				parallelism		=> neuronParallelism,
				weightParallelism	=> weightParallelism,

				-- excitatory spikes
				input_parallelism	=> N_inputs,
				N_exc_cnt		=> N_exc_cnt,

				-- inhibitory spikes
				layer_size		=> N_neurons,
				N_inh_cnt		=> N_inh_cnt,

				-- elaboration steps
				N_cycles_cnt		=> N_cycles_cnt,

				-- exponential decay shift
				shift			=> shift
					
			)

			port map(
				-- input
				clk			=> clk,
				rst_n			=> rst_n,
				start			=> start,
				input_spikes		=> input_spikes,
							

				-- input parameters
				v_th_value		=> v_th_value,
				v_reset			=> v_reset_sig,
				inh_weight		=> inh_weight_sig,
				exc_weights		=> exc_weights,
							

				-- number of inputs, neurons and cycles
				N_inputs		=> N_inputs_sig,
				N_neurons		=> N_neurons_sig,
				N_cycles		=> N_cycles_sig,
							

				-- output
				exc_cnt			=> exc_cnt,
							
				out_spikes		=> outSpikes,
							
				sample			=> sample,
				ready			=> ready
			);




end architecture behaviour;
