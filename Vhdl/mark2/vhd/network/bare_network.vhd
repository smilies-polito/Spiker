library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bare_network is

	port(
		-- input
		clk			: in std_logic;
		rst_n			: in std_logic;	
		start			: in std_logic;	
		input_spikes		: in std_logic_vector(783 downto 0);

		-- load weights inputs
		weights_wr_en		: in std_logic;
		weights_values		: in signed(16*2-1 downto 0);

		-- output
		out_spikes		: out std_logic_vector(1 downto 0);
		sample			: out std_logic;
		ready			: out std_logic
	);


end bare_network;


architecture behaviour of bare_network is

	-- internal parallelism
	constant parallelism		: integer := 16;
	constant input_parallelism	: integer := 784;
	constant N_exc_cnt		: integer := 11;
	constant layer_size		: integer := 2;
	constant N_inh_cnt		: integer := 2;
	constant total_cycles		: integer := 3500;
	constant N_cycles_cnt		: integer := 12;
	constant shift			: integer := 10;


	-- model parameters
	constant v_th_0_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^10 rounded	
	constant inh_weight_int	: integer 	:= -15*(2**10);	

	-- internal constants
	signal v_th_0		: signed(parallelism-1 downto 0);
	signal v_reset		: signed(parallelism-1 downto 0);
	signal v_th_plus	: signed(parallelism-1 downto 0);
	signal inh_weight	: signed(parallelism-1 downto 0);
	signal N_inputs		: std_logic_vector(N_exc_cnt-1 downto 0);
	signal N_neurons	: std_logic_vector(N_inh_cnt-1 downto 0);
	signal N_cycles		: std_logic_vector(N_cycles_cnt-1 downto 0);

	-- signal between layer and synapses
	signal exc_cnt_address	: std_logic_vector(N_exc_cnt-1 downto 0);
	signal exc_weights	: signed(parallelism*layer_size-1 downto 0);



	component layer is

		generic(

			-- internal parallelism
			parallelism		: integer := 16;

			-- excitatory spikes
			input_parallelism	: integer := 8;
			N_exc_cnt		: integer := 3;

			-- inhibitory spikes
			layer_size		: integer := 4;
			N_inh_cnt		: integer := 2;

			-- elaboration steps
			N_cycles_cnt		: integer := 4;

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
			v_th_0			: in signed(parallelism-1 downto 0);		
			v_reset			: in signed(parallelism-1 downto 0);	
			inh_weight		: in signed(parallelism-1 downto 0);		
			v_th_plus		: in signed(parallelism-1 downto 0);	
			exc_weights		: in signed
						(layer_size*parallelism-1 downto 0);

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



	component generic_reg_file_signed is

		generic(
		
			N_registers	: integer := 4;

			N_address	: integer := 2;

			N		: integer := 8

		);

		port(
			-- input
			clk		: in std_logic;
			address		: in std_logic_vector(N_address-1 downto 0);
			wr_en		: in std_logic;
			input_data	: in signed(N-1 downto 0); 
			
			-- output
			output_data	: out signed(N-1 downto 0)
		);

	end component generic_reg_file_signed; 

begin

	v_th_0		<= to_signed(v_th_0_int, parallelism);
	v_reset		<= to_signed(v_reset_int, parallelism);
	v_th_plus	<= to_signed(v_th_plus_int, parallelism);
	inh_weight	<= to_signed(inh_weight_int, parallelism);
	N_inputs	<= std_logic_vector(to_signed(input_parallelism,
						N_exc_cnt));
	N_neurons	<= std_logic_vector(to_signed(layer_size, N_inh_cnt));
	N_cycles	<= std_logic_vector(to_signed(total_cycles,
						N_cycles_cnt));
	
	

	single_layer	: layer 

		generic map(

			-- internal parallelism
			parallelism		=> parallelism,

			-- excitatory spikes
			input_parallelism	=> input_parallelism,
			N_exc_cnt		=> N_exc_cnt,

			-- inhibitory spikes
			layer_size		=> layer_size,
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
			v_th_0			=> v_th_0,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			v_th_plus		=> v_th_plus,
			exc_weights		=> exc_weights,
						
			-- number of inputs, neurons and cycles
			N_inputs		=> N_inputs,	
			N_neurons		=> N_neurons,
			N_cycles		=> N_cycles,	
			
			-- output
			exc_cnt			=> exc_cnt_address,
			out_spikes		=> out_spikes,
			sample			=> sample,
			ready			=> ready
		);


	synapses	: for i in 0 to layer_size-1
	generate

		single_neuron_synapses	: generic_reg_file_signed

			generic map(
			
				N_registers	=> input_parallelism,
				N_address	=> N_exc_cnt,
				N		=> parallelism

			)

			port map(
				-- input
				clk		=> clk,
				address		=> exc_cnt_address,
				wr_en		=> weights_wr_en,
				input_data	=>
			       		weights_values((i+1)*parallelism-1 downto
					i*parallelism),
				
				-- output
				output_data	=>
					exc_weights((i+1)*parallelism-1 downto
					i*parallelism)
			);

	end generate synapses;




end architecture behaviour;
