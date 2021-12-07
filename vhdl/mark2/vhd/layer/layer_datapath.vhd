library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity layer_datapath is

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
		-- control input
		input_spikes		: in std_logic_vector
						(input_parallelism-1 downto 0);
		clk			: in std_logic;
		exc_en			: in std_logic;	
		anticipate_exc		: in std_logic;	
		inh_en			: in std_logic;	
		anticipate_inh		: in std_logic;	
		exc_cnt_rst_n		: in std_logic;	
		exc_cnt_en		: in std_logic;	
		inh_cnt_rst_n		: in std_logic;	
		inh_cnt_en		: in std_logic;	
		exc_or_inh_sel		: in std_logic;	
		inh			: in std_logic;	
		cycles_cnt_rst_n	: in std_logic;	
		cycles_cnt_en		: in std_logic;	
		rst_n			: in std_logic;	
		start			: in std_logic;	

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

		-- control output
		exc_or			: out std_logic;
		exc_stop		: out std_logic;
		inh_or			: out std_logic;
		inh_stop		: out std_logic;
		stop			: out std_logic;

		-- output
		out_spikes		: out std_logic_vector
						(layer_size-1 downto 0);
		neurons_ready		: out std_logic;
		exc_cnt			: out std_logic_vector
						(N_exc_cnt-1 downto 0)
	
	);

end entity layer_datapath;


architecture behaviour of layer_datapath is

	signal exc_spikes		: std_logic_vector(input_parallelism-1 downto 0);
	signal inh_spikes		: std_logic_vector(layer_size-1 downto 0);
	signal exc_spike		: std_logic;
	signal inh_spike		: std_logic;
	signal inh_cnt			: std_logic_vector(N_inh_cnt-1 downto 0);
	signal cycles			: std_logic_vector(N_cycles_cnt-1 downto 0);
	signal input_spike		: std_logic;
	signal exc_or_internal		: std_logic;
	signal exc_stop_internal	: std_logic;
	signal inh_or_internal		: std_logic;
	signal inh_stop_internal	: std_logic;
	signal stop_internal		: std_logic;
	signal feedback_spikes		: std_logic_vector(layer_size-1 downto 0);
	signal spikes			: std_logic_vector(2**N_inh_cnt-1 downto 0);


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
			input_bits		: in std_logic_vector(N_bit-1 downto 0);
			select_cnt_en		: in std_logic;
			select_cnt_rst_n	: in std_logic;
			N_inputs		: in std_logic_vector(N_cnt-1 downto 0);

			-- output
			all_inputs		: out std_logic;
			selected_input		: out std_logic;
			input_index		: out std_logic_vector(N_cnt-1 downto 0);
			stop			: out std_logic		
		);

	end component bit_selection;


	component cnt is

		generic(
			N		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(N-1 downto 0)		
		);

	end component cnt;


	component cmp_eq is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			in0	: in std_logic_vector(N-1 downto 0);
			in1	: in std_logic_vector(N-1 downto 0);

			-- output
			cmp_out	: out std_logic
		);

	end component cmp_eq;


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


	component bitMask is

		generic(
			N_cnt		: integer :=3
		);

		port(
			-- input
			input_cnt	: in std_logic_vector(N_cnt-1 downto 0);
			inh		: in std_logic;
			input_bit	: in std_logic;

			-- output
			output_bits	: out std_logic_vector(2**N_cnt-1 downto 0)
		);

	end component bitMask;




	component neurons_layer is

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
			stop		: in std_logic;
			exc_or		: in std_logic;
			exc_stop	: in std_logic;
			inh_or		: in std_logic;
			inh_stop	: in std_logic;
			input_spikes	: in std_logic_vector(layer_size-1 downto 0);

			-- input parameters
			v_th_0		: in signed(N-1 downto 0);		
			v_reset		: in signed(N-1 downto 0);		
			inh_weight	: in signed(N-1 downto 0);		
			v_th_plus	: in signed(N-1 downto 0);		
			exc_weights	: in signed(layer_size*N-1 downto 0);

			-- output
			out_spikes	: out std_logic_vector(layer_size-1 downto 0);
			neurons_ready	: out std_logic
		);

	end component neurons_layer;



begin

	out_spikes	<= feedback_spikes;
	exc_or		<= exc_or_internal;
	exc_stop	<= exc_stop_internal;
	inh_or		<= inh_or_internal;
	inh_stop	<= inh_stop_internal;
	stop		<= stop_internal;


	anticipate_exc_spikes	: anticipate_bits
		generic map(
			-- parallelism
			N		=> input_parallelism		
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
			N		=> layer_size	
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
			N_bit			=> input_parallelism,

			-- selection counter parallelism
			N_cnt			=> N_exc_cnt
		)

		port map(
			-- input
			clk			=> clk,
			input_bits		=> exc_spikes,
			select_cnt_en		=> exc_cnt_en,
			select_cnt_rst_n	=> exc_cnt_rst_n,
			N_inputs		=> N_inputs,

			-- output
			all_inputs		=> exc_or_internal,
			selected_input		=> exc_spike,
			input_index		=> exc_cnt,
			stop			=> exc_stop_internal
		);


	select_inh_spike	: bit_selection 
		generic map(
			-- number of input bits
			N_bit			=> layer_size,

			-- selection counter parallelism
			N_cnt			=> N_inh_cnt
		)

		port map(
			-- input
			clk			=> clk,
			input_bits		=> inh_spikes,
			select_cnt_en		=> inh_cnt_en,
			select_cnt_rst_n	=> inh_cnt_rst_n,
			N_inputs		=> N_neurons,

			-- output
			all_inputs		=> inh_or_internal,
			selected_input		=> inh_spike,
			input_index		=> inh_cnt,
			stop			=> inh_stop_internal
		);


	cycles_cnt		: cnt
		generic map(
			N			=> N_cycles_cnt
		)

		port map(
			-- input
			clk			=> clk,
			cnt_en			=> cycles_cnt_en,
			cnt_rst_n		=> cycles_cnt_rst_n,

			-- output
			cnt_out			=> cycles
		);


	cycles_cmp		: cmp_eq
		generic map(
			N			=> N_cycles_cnt
		)

		port map(
			-- input
			in0			=> cycles,
			in1			=> N_cycles,

			-- output
			cmp_out			=> stop_internal
		);



	exc_or_inh_mux		: mux2to1_std_logic

		port map(	
			-- inputs	
			sel			=> exc_or_inh_sel,
			in0			=> exc_spike,
			in1			=> inh_spike,

			-- output
			mux_out			=> input_spike
		);


	generate_spikes		: bitMask

		generic map(
			N_cnt			=> N_inh_cnt
		)

		port map(
			-- input
			input_cnt		=> inh_cnt,
			inh			=> inh,
			input_bit		=> input_spike,

			-- output
			output_bits		=> spikes
		);



	bare_layer : neurons_layer

		generic map(
			-- parallelism
			N		=> parallelism,	

			-- number of neurons in the layer
			layer_size	=> layer_size,

			-- shift amount
			shift		=> shift
		)

		port map(
			-- input controls
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop_internal,
			exc_or	       	=> exc_or_internal,
			exc_stop       	=> exc_stop_internal,
			inh_or	        => inh_or_internal,
			inh_stop        => inh_stop_internal,
                       	input_spikes	=> spikes(layer_size-1 downto 0),

			-- input parameters
			v_th_0		=> v_th_0,
			v_reset		=> v_reset,
			inh_weight	=> inh_weight,
			exc_weights	=> exc_weights,
			v_th_plus	=> v_th_plus,
                                                       
			-- output          
			out_spikes	=> feedback_spikes,
			neurons_ready	=> neurons_ready
		);


end architecture behaviour;
