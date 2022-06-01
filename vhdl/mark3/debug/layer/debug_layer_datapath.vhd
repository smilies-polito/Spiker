library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity debug_layer_datapath is

	generic(

		-- int parallelism
		parallelism		: integer := 16;
		weightsParallelism	: integer := 5;

		-- input spikes
		N_inputs		: integer := 784;

		-- must be one bit larger that the parallelism required to count
		-- up to N_inputs
		N_inputs_cnt		: integer := 11;

		-- inhibitory spikes
		N_neurons		: integer := 400;

		-- must be one bit larger that the parallelism required to count
		-- up to N_neurons
		N_neurons_cnt		: integer := 10;

		-- exponential decay shift
		shift			: integer := 10
			
	);

	port(
		-- control input
		clk			: in std_logic;
		exc_en			: in std_logic;	
		anticipate_exc		: in std_logic;	
		inh_en			: in std_logic;	
		anticipate_inh		: in std_logic;	
		exc_cnt_en		: in std_logic;	
		exc_cnt_rst_n		: in std_logic;	
		inh_cnt_en		: in std_logic;	
		inh_cnt_rst_n		: in std_logic;	
		exc_or_inh_sel		: in std_logic;	
		init_v_th		: in std_logic;
		rst_n			: in std_logic;	
		start			: in std_logic;	
		stop			: in std_logic;	
		inh			: in std_logic;	

		-- address to select the neurons
		v_th_addr		: in std_logic_vector(N_neurons_cnt-1
						  downto 0);

		-- data input
		input_spikes		: in std_logic_vector
						(N_inputs-1 downto 0);

		-- input parameters
		v_th_value		: in signed(parallelism-1 downto 0);		
		v_reset			: in signed(parallelism-1 downto 0);	
		inh_weight		: in signed(parallelism-1 downto 0);		
		exc_weights		: in signed
						(N_neurons*weightsParallelism-1
						 downto 0);

		-- terminal counters 
		N_inputs_tc		: in std_logic_vector
						(N_inputs_cnt-1 downto 0);
		N_neurons_tc		: in std_logic_vector
						(N_neurons_cnt-1 downto 0);

		-- control output
		exc_or			: out std_logic;
		exc_stop		: out std_logic;
		inh_or			: out std_logic;
		inh_stop		: out std_logic;

		-- output
		out_spikes		: out std_logic_vector
						(N_neurons-1 downto 0);
		all_ready		: out std_logic;

		-- output address to select the excitatory weights
		exc_cnt			: out std_logic_vector
						(N_inputs_cnt-1 downto 0);

		-- debug output
		v_out			: out signed(parallelism-1
						downto 0)
	
	);

end entity debug_layer_datapath;


architecture behaviour of debug_layer_datapath is

	signal exc_spikes		: std_logic_vector(N_inputs-1 downto 0);
	signal inh_spikes		: std_logic_vector(N_neurons-1 
						downto 0);
	signal exc_spike		: std_logic;
	signal inh_spike		: std_logic;
	signal inh_cnt			: std_logic_vector(N_neurons_cnt-1 
						downto 0);
	signal spike			: std_logic;
	signal exc_or_int		: std_logic;
	signal exc_stop_int		: std_logic;
	signal inh_or_int		: std_logic;
	signal inh_stop_int		: std_logic;
	signal feedback_spikes		: std_logic_vector(N_neurons-1 
						downto 0);
	signal neuron_addr		: std_logic_vector(N_neurons_cnt-1
						downto 0);


	component anticipate_bits is

		generic(
			-- parallelism
			N		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			bits_en		: in std_logic;
			anticipate	: in std_logic;
			input_bits	: in std_logic_vector(N-1 downto 0);

			-- output
			output_bits	: out std_logic_vector(N-1 downto 0)	
		);

	end component anticipate_bits;


	component bit_selection is

		generic(
			-- number of input bits
			N_bit			: integer := 8;

			-- selection counter parallelism
			N_cnt			: integer := 3		
		);

		port(
			-- input
			clk			: in std_logic;
			input_bits		: in std_logic_vector(N_bit-1 
							  downto 0);
			select_cnt_en		: in std_logic;
			select_cnt_rst_n	: in std_logic;
			N_inputs		: in std_logic_vector(N_cnt-1 
							  downto 0);

			-- output
			all_inputs		: out std_logic;
			selected_input		: out std_logic;
			input_index		: out std_logic_vector(N_cnt-1
							  downto 0);
			stop			: out std_logic		
		);

	end component bit_selection;



	component mux2to1_std_logic is
		port(	
			-- inputs	
			sel	: in std_logic;
			in0	: in std_logic;
			in1	: in std_logic;

			-- output
			mux_out	: out std_logic
		);

	end component mux2to1_std_logic;



	component mux2to1 is

		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			sel	: in std_logic;
			in0	: in std_logic_vector(N-1 downto 0);
			in1	: in std_logic_vector(N-1 downto 0);

			-- output
			mux_out	: out std_logic_vector(N-1 downto 0)
		);

	end component mux2to1;



	component debug_complete_neurons is

		generic(
			-- int parallelism_
			parallelism		: integer := 16;
			weightsParallelism	: integer := 5;

			-- number of neurons in the layer
			N_neurons		: integer := 400;
			N_addr			: integer := 9;

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
			inh			: in std_logic;
			load_v_th		: in std_logic;
			neuron_addr		: in std_logic_vector(N_addr-1
							  downto 0);

			-- input
			input_spike		: in std_logic;

			-- input parameters
			v_th_value		: in signed(parallelism-1 
							downto 0);		
			v_reset			: in signed(parallelism-1 
							downto 0);		
			inh_weight		: in signed(parallelism-1 
							downto 0);		
			exc_weights		: in signed(N_neurons*
							weightsParallelism-1
							downto 0);

			-- output
			out_spikes		: out std_logic_vector
							(N_neurons-1 downto 0);
			all_ready		: out std_logic;

			-- debug output
			v_out			: out signed(parallelism-1
							downto 0)
		);
		
	end component debug_complete_neurons;


begin

	out_spikes	<= feedback_spikes;
	exc_or		<= exc_or_int;
	exc_stop	<= exc_stop_int;
	inh_or		<= inh_or_int;
	inh_stop	<= inh_stop_int;


	anticipate_exc_spikes	: anticipate_bits
		generic map(
			-- parallelism
			N		=> N_inputs		
		)

		port map(
			-- input
			clk		=> clk,
			bits_en		=> exc_en,
			anticipate	=> anticipate_exc,
			input_bits	=> input_spikes,

			-- output
			output_bits	=> exc_spikes
		);



	anticipate_inh_spikes	: anticipate_bits
		generic map(
			-- parallelism
			N		=> N_neurons	
		)

		port map(
			-- input
			clk		=> clk,
			bits_en		=> inh_en,
			anticipate	=> anticipate_inh,
			input_bits	=> feedback_spikes,

			-- output
			output_bits	=> inh_spikes
		);



	select_exc_spike	: bit_selection 
		generic map(
			-- number of input bits
			N_bit			=> N_inputs,

			-- selection counter parallelism
			N_cnt			=> N_inputs_cnt
		)

		port map(
			-- input
			clk			=> clk,
			input_bits		=> exc_spikes,
			select_cnt_en		=> exc_cnt_en,
			select_cnt_rst_n	=> exc_cnt_rst_n,
			N_inputs		=> N_inputs_tc,

			-- output
			all_inputs		=> exc_or_int,
			selected_input		=> exc_spike,
			input_index		=> exc_cnt,
			stop			=> exc_stop_int
		);


	select_inh_spike	: bit_selection 
		generic map(
			-- number of input bits
			N_bit			=> N_neurons,

			-- selection counter parallelism
			N_cnt			=> N_neurons_cnt
		)

		port map(
			-- input
			clk			=> clk,
			input_bits		=> inh_spikes,
			select_cnt_en		=> inh_cnt_en,
			select_cnt_rst_n	=> inh_cnt_rst_n,
			N_inputs		=> N_neurons_tc,

			-- output
			all_inputs		=> inh_or_int,
			selected_input		=> inh_spike,
			input_index		=> inh_cnt,
			stop			=> inh_stop_int
		);



	exc_or_inh_mux		: mux2to1_std_logic

		port map(	
			-- inputs	
			sel			=> exc_or_inh_sel,
			in0			=> exc_spike,
			in1			=> inh_spike,

			-- output
			mux_out			=> spike
		);


	addr_mux		:  mux2to1

		generic map(
			N			=> N_neurons_cnt		
		)

		port map(
			-- inputs
			sel			=> init_v_th,
			in0			=> inh_cnt,
			in1			=> v_th_addr,

			-- output
			mux_out			=> neuron_addr
		);



	bare_layer : debug_complete_neurons

		generic map(

			-- parallelism
			parallelism		=> parallelism,	
			weightsParallelism	=> weightsParallelism,

			-- number of neurons in the layer
			N_neurons		=> N_neurons,
			N_addr			=> N_neurons_cnt,

			-- shift amount
			shift			=> shift
		)

		port map(
			-- input controls
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			stop			=> stop,
			exc_or			=> exc_or_int,
			exc_stop		=> exc_stop_int,
			inh_or			=> inh_or_int,
			inh_stop		=> inh_stop_int,
			inh			=> inh,
			load_v_th		=> init_v_th,
			neuron_addr		=> neuron_addr,

			-- input
                       	input_spike		=> spike,

			-- input parameters
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			exc_weights		=> exc_weights,
							       
			-- output		   
			out_spikes		=> feedback_spikes,
			all_ready		=> all_ready,

			-- debug output
			v_out			=> v_out
		);



end architecture behaviour;
