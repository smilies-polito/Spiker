library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity layer_cu_dp_tb is
end entity layer_cu_dp_tb;


architecture test of layer_cu_dp_tb is

	-- internal parallelism
	constant parallelism		: integer := 16;

	-- excitatory spikes
	constant input_parallelism	: integer := 2;
	constant N_exc_cnt		: integer := 3;

	-- inhibitory spikes
	constant layer_size		: integer := 2;
	constant N_inh_cnt		: integer := 2;

	-- elaboration steps
	constant N_cycles_cnt		: integer := 4;

	-- exponential decay shift
	constant shift			: integer := 1;

	-- model parameters
	constant v_th_0_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^11 rounded	
	constant inh_weight_int	: integer 	:= 7*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);


	-- number of inputs, neurons and cycles
	signal N_inputs			: std_logic_vector
						(N_exc_cnt-1 downto 0);
	signal N_neurons		: std_logic_vector
						(N_inh_cnt-1 downto 0);
	signal N_cycles			: std_logic_vector
						(N_cycles_cnt-1 downto 0);
	-- input parameters
	signal v_th_0		: signed(parallelism-1 downto 0);
	signal v_reset		: signed(parallelism-1 downto 0);
	signal v_th_plus	: signed(parallelism-1 downto 0);
	signal inh_weight	: signed(parallelism-1 downto 0);
	signal exc_weights	: signed(layer_size*parallelism-1 downto 0);

	-- input
	signal clk			: std_logic;
	signal rst_n			: std_logic;
	signal start			: std_logic;
	signal input_spikes		: std_logic_vector
						(input_parallelism-1 downto 0);

	-- signals from datapath
	signal exc_or			: std_logic;		
	signal exc_stop			: std_logic;		
	signal inh_or			: std_logic;		
	signal inh_stop			: std_logic;
	signal stop			: std_logic;		

	-- towards datapath
	signal exc_en			: std_logic;	
	signal anticipate_exc		: std_logic;	
	signal inh_en			: std_logic;	
	signal anticipate_inh		: std_logic;	
	signal exc_cnt_rst_n		: std_logic;	
	signal exc_cnt_en		: std_logic;	
	signal inh_cnt_rst_n		: std_logic;	
	signal inh_cnt_en		: std_logic;	
	signal exc_or_inh_sel		: std_logic;	
	signal inh			: std_logic;	
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;	

	-- output
	signal sample			: std_logic;
	signal layer_ready		: std_logic;
	signal out_spikes		: std_logic_vector(layer_size-1 downto 0);
	signal exc_cnt			: std_logic_vector
						(N_exc_cnt-1 downto 0);

	-- control unit internal signals
	type states is(
		reset,
		idle,
		sample_spikes,
		exc_update,
		inh_update
	);

	signal present_state, next_state	: states;

	-- datapath internal signals
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
	signal neurons_ready		: std_logic;



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


	-- model parameters binary conversion
	v_th_0		<= to_signed(v_th_0_int, parallelism);
	v_reset		<= to_signed(v_reset_int, parallelism);
	v_th_plus	<= to_signed(v_th_plus_int, parallelism);
	inh_weight	<= to_signed(inh_weight_int, parallelism);

	exc_weights_init	: process
	begin
		init	: for i in 0 to layer_size-1
		loop
			exc_weights((i+1)*parallelism-1 downto i*parallelism) <= 
					to_signed(exc_weight_int, parallelism);
		end loop init;

		wait;

	end process exc_weights_init;








	-- clock
	clock_gen : process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- reset
	reset_gen : process
	begin
		rst_n	<= '1';		-- 0 ns
		wait for 14 ns;
		rst_n	<= '0';		-- 14 ns
		wait for 3 ns;
		rst_n	<= '1';		-- 17 ns
		wait;
	end process reset_gen;



	-- start
	start_gen : process
	begin
		start	<= '0';		-- 0 ns
		wait for 38 ns;
		start	<= '1';		-- 38 ns
		wait for 12 ns;
		start	<= '0';		-- 50 ns
		wait;
	end process start_gen;


	-- input_spikes
	input_spikes_gen: process
	begin
		input_spikes	<= "00"; -- 0 ns	
		wait for 62 ns;
		input_spikes	<= "11"; -- 62 ns
		wait for 48 ns;
		input_spikes	<= "00"; -- 110 ns
		wait for 48 ns;
		input_spikes	<= "01"; -- 158 ns
		wait for 84 ns;
		input_spikes	<= "00"; -- 242 ns
		wait;
	end process input_spikes_gen;










	-- state transition
	state_transition	: process(clk, rst_n)
	begin

		if rst_n = '0'
		then
			present_state	<= reset;

		elsif clk'event and clk = '1'
		then
			present_state	<= next_state;
		end if;

	end process state_transition;


	-- state evaluation
	state_evaluation	: process(present_state, start, stop, exc_or,
					exc_stop, inh_or, inh_stop)
	begin

		-- default case
		next_state	<= reset;

		case present_state is

			-- reset
			when reset =>

				next_state <= idle;


			-- idle
			when idle =>

				if start = '1'
				then
					next_state <= sample_spikes;
				else
					next_state <= idle;
				end if;


			-- sample_spikes
			when sample_spikes =>

				if stop = '1'
				then
					next_state <= idle;
				else
					if exc_or = '1'
					then
						next_state <= exc_update;

					elsif inh_or = '1'
					then
						next_state <= inh_update;
					
					else
						next_state <= sample_spikes;
					end if;
				end if;


			-- exc_update
			when exc_update =>

				if exc_stop = '0'
				then
					next_state <= exc_update;

				elsif inh_or = '1'
				then
					next_state <= inh_update;

				else
					next_state <= sample_spikes;
				end if;


			-- inh_update
			when inh_update =>
				
				if inh_stop = '1'
				then
					next_state <= sample_spikes;
				else
					next_state <= inh_update;
				end if;


			when others =>

				next_state <= reset;		

		end case;

	end process state_evaluation;




	output_evaluation	: process(present_state)
	begin

		-- default values
		exc_en			<= '0';
		anticipate_exc		<= '0';
		inh_en			<= '0';
		anticipate_inh		<= '0';
		exc_cnt_en		<= '0';
		exc_cnt_rst_n		<= '0';
		inh_cnt_en		<= '0';
		inh_cnt_rst_n		<= '0';
		exc_or_inh_sel		<= '0';
		inh			<= '0';
		layer_ready		<= '0';
		sample			<= '0';
		cycles_cnt_en		<= '0';
		cycles_cnt_rst_n	<= '1';


		case present_state is

			-- reset
			when reset =>

				cycles_cnt_rst_n	<= '0';

			-- idle
			when idle =>

				layer_ready		<= '1';
				cycles_cnt_rst_n	<= '0';

			-- sample_spikes
			when sample_spikes =>

				sample			<= '1';
				exc_en			<= '1';
				anticipate_exc		<= '1';
				inh_en			<= '1';
				anticipate_inh		<= '1';
				cycles_cnt_en		<= '1';

			-- exc_update
			when exc_update =>

				exc_cnt_en		<= '1';
				exc_cnt_rst_n		<= '1';


			-- inh_update
			when inh_update =>

				inh_cnt_en		<= '1';
				inh_cnt_rst_n		<= '1';
				exc_or_inh_sel		<= '1';
				inh			<= '1';

			when others =>

				cycles_cnt_rst_n	<= '0';

		end case;

	end process output_evaluation;




	


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




end architecture test;
