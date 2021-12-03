library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity layer_datapath is
end entity layer_datapath;


architecture behaviour of layer_datapath is


	-- internal parallelism
	constant parallelism		: integer := 8;

	-- excitatory spikes
	constant input_parallelism	: integer := 4;
	constant N_exc_cnt		: integer := 2;

	-- inhibitory spikes
	constant layer_size		: integer := 4;
	constant N_inh_cnt		: integer := 2;

	-- elaboration steps
	constant N_cycles_cnt		: integer := 4;

	-- exponential decay shift
	constant shift			: integer := 1;
	
	-- control input
	signal input_spikes		: std_logic_vector
	 				     (input_parallelism-1 downto 0);
	signal clk			: std_logic;
	signal exc_en			: std_logic;	
	signal anticipate_exc		: std_logic;	
	signal inh_en			: std_logic;	
	signal anticipate_inh		: std_logic;	
	signal exc_cnt_rst_n		: std_logic;	
	signal exc_cnt_en		: std_logic;	
	signal inh_cnt_rst_n		: std_logic;	
	signal inh_cnt_en		: std_logic;	
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;	
	signal exc_or_inh_sel		: std_logic;	
	signal inh_elaboration		: std_logic;	
	signal rest_en			: std_logic;	
	signal rst_n			: std_logic;	
	signal start			: std_logic;	
	signal mask1			: std_logic;	
	signal mask2			: std_logic;

	-- input parameters
	signal v_th_0			: signed(parallelism-1 downto 0);		
	signal v_reset			: signed(parallelism-1 downto 0);	
	signal inh_weight		: signed(parallelism-1 downto 0);		
	signal v_th_plus		: signed(parallelism-1 downto 0);	
	signal exc_weights		: signed
						(layer_size*parallelism-1 downto 0);

	-- number of inputs, neurons and cycles
	signal N_inputs			: std_logic_vector
					     (N_exc_cnt-1 downto 0);
	signal N_neurons		: std_logic_vector
	 				     (layer_size-1 downto 0);
	signal N_cycles			: std_logic_vector
						(N_cycles_cnt-1 downto 0);

	-- control output
	signal exc_or			: std_logic;
	signal exc_stop			: std_logic;
	signal inh_or			: std_logic;
	signal inh_stop			: std_logic;
	signal stop			: std_logic;

	-- output
	signal out_spikes			: std_logic_vector
					    (layer_size-1 downto 0);
	signal layer_ready			: std_logic;
	signal exc_cnt				: std_logic_vector;



	-- internal signals
	signal exc_spikes	: std_logic_vector(input_parallelism-1 downto 0);
	signal inh_spikes	: std_logic_vector(layer_size-1 downto 0);
	signal exc_spike	: std_logic;
	signal inh_spike	: std_logic;
	signal inh_cnt		: std_logic_vector(N_inh_cnt-1 downto 0);
	signal cycles		: std_logic_vector(N_cycles_cnt-1 downto 0);
	signal input_spike	: std_logic;
	signal exc_or_internal	: std_logic;
	signal inh_or_internal	: std_logic;
	signal feedback_spikes	: std_logic_vector(layer_size-1 downto 0);
	signal start1		: std_logic;
	signal start2		: std_logic;


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
			start1		: in std_logic;		
			start2		: in std_logic;		
			rest_en		: in std_logic;
			input_spike	: in std_logic;
			neuron_cnt	: in std_logic_vector(N_cnt-1 downto 0);
			inh_elaboration	: in std_logic;

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

	end component neurons_layer;



begin


	-- clock
	clock_gen 		: process
	begin
		clk	<= '0';			-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         	-- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- reset
	reset_gen 		: process
	begin
		rst_n	<= '1';			-- 0 ns
		wait for 14 ns;
		rst_n	<= '0';			-- 14 ns
		wait for 3 ns;
		rst_n	<= '1';			-- 17 ns
		wait;
	end process reset_gen;


	-- start
	start_gen 		: process
	begin
		start	<= '0';			-- 0 ns
		wait for 38 ns;
		start	<= '1';			-- 38 ns
		wait for 12 ns;
		start	<= '0';			-- 50 ns
		wait;
	end process start_gen;


	-- input_spikes
	input_spikes_gen	: process
	begin
		wait for 50 ns;
		input_spikes	<= "0000";
		wait for 24 ns;
		input_spikes	<= "0010";
		wait;
	end process input_spikes_gen;


	-- exc_en
	exc_en_gen 		: process
	begin
		exc_en	<= '0';			-- 0 ns
		wait for 50 ns;			                    
		exc_en	<= '1';         	-- 50 ns
		wait for 36 ns;	
		exc_en	<= '0';			-- 86 ns
		wait;		
	end process exc_en_gen;


	
	-- anticipate_exc
	anticipate_exc_gen 	: process
	begin
		anticipate_exc	<= '0';			-- 0 ns
		wait for 50 ns;			                    
		anticipate_exc	<= '1';         	-- 50 ns
		wait for 36 ns;	
		anticipate_exc	<= '0';			-- 86 ns
		wait;		
	end process anticipate_exc_gen;


	-- mask1
	mask1_gen 	: process
	begin
		mask1	<= '0';			-- 0 ns
		wait for 50 ns;			                    
		mask1	<= '1';         	-- 50 ns
		wait for 36 ns;	
		mask1	<= '0';			-- 86 ns
		wait;		
	end process mask1_gen;

	
	-- mask2
	mask2_gen 	: process
	begin
		mask2	<= '0';			-- 0 ns
		wait for 50 ns;			                    
		mask2	<= '1';         	-- 50 ns
		wait for 36 ns;	
		mask2	<= '0';			-- 86 ns
		wait;		
	end process mask2_gen;


	-- inh_en
	inh_en_gen 	: process
	begin
		inh_en	<= '0';			-- 0 ns
		wait for 50 ns;			                    
		inh_en	<= '1';         	-- 50 ns
		wait for 36 ns;	
		inh_en	<= '0';			-- 86 ns
		wait;		
	end process inh_en_gen;


	-- anticipate_inh
	anticipate_inh_gen 	: process
	begin
		anticipate_inh	<= '0';		-- 0 ns
		wait for 50 ns;			                    
		anticipate_inh	<= '1';         -- 50 ns
		wait for 36 ns;	
		anticipate_inh	<= '0';		-- 86 ns
		wait;		
	end process anticipate_inh_gen;

















	out_spikes	<= feedback_spikes;

	start1		<= exc_or_internal and mask1;
	start2		<= inh_or_internal and mask2;

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
			all_inputs		=> exc_or,
			selected_input		=> exc_spike,
			input_index		=> exc_cnt,
			stop			=> exc_stop
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
			all_inputs		=> inh_or,
			selected_input		=> inh_spike,
			input_index		=> inh_cnt,
			stop			=> inh_stop
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
			cmp_out			=> stop
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




	bare_layer		: neurons_layer
		generic map(
			-- neurons counter parallelism
			N_cnt			=> N_inh_cnt,

			-- internal parallelism
			N			=> parallelism,

			-- number of neurons in the layer
			layer_size		=> layer_size,

			-- shift during the exponential decay
			shift			=> shift
		)

		port map(
			-- control input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			start1			=> start1,
			start2			=> start2,
			rest_en			=> rest_en,
			input_spike		=> input_spike,
			neuron_cnt		=> inh_cnt,
			inh_elaboration		=> inh_elaboration,

			-- input parameters
			v_th_0		 	=> v_th_0,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			v_th_plus		=> v_th_plus,
			exc_weights		=> exc_weights,

			-- output
			out_spikes		=> feedback_spikes,
			layer_ready		=> layer_ready
		);



end architecture behaviour;
