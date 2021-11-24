library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity layer_cu_dp_tb is
end entity layer_cu_dp_tb;


architecture test of layer_cu_dp_tb is


	-- internal parallelism
	constant parallelism		: integer := 16;

	-- excitatory spikes
	constant input_parallelism	: integer := 4;
	constant N_exc_cnt		: integer := 2;

	-- inhibitory spikes
	constant layer_size		: integer := 4;
	constant N_inh_cnt		: integer := 2;

	-- elaboration steps
	constant cycles_number		: integer := 50;
	constant N_cycles_cnt		: integer := 6;

	-- exponential decay shift
	constant shift			: integer := 1;


	-- model parameters
	constant v_th_0_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^11 rounded	
	constant inh_weight_int	: integer 	:= -15*(2**10);	
	constant exc_weight_int	: integer 	:= 4*(2**10);



	-- common inputs
	signal clk			: std_logic;
	signal rst_n			: std_logic;
	signal start			: std_logic;

	-- input parameters
	signal v_th_0			: signed(parallelism-1 downto 0);		
	signal v_reset			: signed(parallelism-1 downto 0);	
	signal inh_weight		: signed(parallelism-1 downto 0);		
	signal v_th_plus		: signed(parallelism-1 downto 0);	
	signal exc_weights		: signed(layer_size*
						parallelism-1 downto 0);

	-- datapath input
	signal input_spikes		: std_logic_vector(input_parallelism-1
						downto 0);

	-- number of inputs, neurons and cycles
	signal N_inputs			: std_logic_vector
					     (N_exc_cnt-1 downto 0);
	signal N_neurons			: std_logic_vector
					     (N_inh_cnt-1 downto 0);
	signal N_cycles			: std_logic_vector
						(N_cycles_cnt-1 downto 0);


	-- from datapath towards control unit
	signal stop			: std_logic;		
	signal exc_or			: std_logic;		
	signal exc_stop			: std_logic;		
	signal inh_or			: std_logic;		
	signal inh_stop			: std_logic;

	-- from control unit towards datapath
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
	signal mask1			: std_logic;	
	signal mask2			: std_logic;

	-- control unit output
	signal sample			: std_logic;
	signal snn_ready		: std_logic;


	-- datapath output
	signal out_spikes		: std_logic_vector
					    (layer_size-1 downto 0);
	signal layer_ready		: std_logic;
	signal exc_cnt			: std_logic_vector
						(N_exc_cnt-1 downto 0);



	-- control unit internal signals
	type states is(
		reset,
		idle,
		sample_spikes,
		exc_update,
		exc_end,
		inh_update,
		inh_end,
		rest		
	);

	signal present_state, next_state	: states;



	-- datapath internal signals
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
			out_spikes	: out std_logic_vector
						(layer_size-1 downto 0);
			layer_ready	: out std_logic
		);

	end component neurons_layer;





begin


	-- model parameters binary conversion
	v_th_0		<= to_signed(v_th_0_int, parallelism);
	v_reset		<= to_signed(v_reset_int, parallelism);
	v_th_plus	<= to_signed(v_th_plus_int, parallelism);
	inh_weight	<= to_signed(inh_weight_int, parallelism);


	N_inputs	<= std_logic_vector(to_unsigned(input_parallelism - 2,
				N_exc_cnt));
	
	N_neurons	<= std_logic_vector(to_unsigned(layer_size - 2,
				N_inh_cnt));

	N_cycles	<= std_logic_vector(to_unsigned(cycles_number - 2,
				N_cycles_cnt));


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
		clk	<= '0';			-- falling edge i*12ns
		wait for 6 ns;						    
		clk	<= '1';			-- rising edge 6ns + i*12ns
		wait for 6 ns;				
	end process clock_gen;




	-- reset
	reset_gen : process
	begin
		rst_n	<= '1';			-- 0 ns
		wait for 14 ns;
		rst_n	<= '0';			-- 14 ns
		wait for 3 ns;
		rst_n	<= '1';			-- 17 ns
		wait;
	end process reset_gen;



	-- start
	start_gen : process
	begin
		start	<= '0';			-- 0 ns
		wait for 38 ns;
		start	<= '1';			-- 38 ns
		wait for 12 ns;
		start	<= '0';			-- 50 ns
		wait;
	end process start_gen;




	-- input_spikes
	input_spikes_gen: process
	begin
		wait for 50 ns;
		input_spikes	<= "0000";	-- 50 ns	
		wait for 24 ns;			
		input_spikes	<= "0010";	-- 74 ns
		wait for 12 ns;			
		input_spikes	<= "1111";	-- 86 ns
		wait for 12 ns;			
		input_spikes	<= "1001";	-- 98 ns
		wait for 12 ns;			
		input_spikes	<= "0110";	-- 110 ns
		wait for 12 ns;			
		input_spikes	<= "0001";	-- 122 ns
		wait for 12 ns;			
		input_spikes	<= "0000";	-- 134 ns
		wait for 12 ns;			
		input_spikes	<= "1111";	-- 146 ns
		wait for 12 ns;			
		input_spikes	<= "1011";	-- 158 ns
		wait for 12 ns;			
		input_spikes	<= "0100";	-- 170 ns
		wait for 12 ns;			
		input_spikes	<= "0000";	-- 182 ns
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

		procedure from_inh_stop_on(

			-- input
			signal inh_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if inh_stop = '1'
			then
				next_state <= inh_end;
			else
				next_state <= inh_update;
			end if;

		end procedure from_inh_stop_on;


		procedure from_inh_or_on(
			
			-- input
			signal inh_or		: in std_logic;
			signal inh_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if inh_or ='1'
			then
				from_inh_stop_on(

					-- input
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state		
				);
			else
				next_state <= sample_spikes;
			end if;

		end procedure from_inh_or_on;


		procedure from_exc_stop_on(

			-- input
			signal exc_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if exc_stop = '1'
			then
				next_state <= exc_end;
			else
				next_state <= exc_update;
			end if;

		end procedure from_exc_stop_on;


		procedure from_exc_or_on(
			
			-- input
			signal exc_or		: in std_logic;
			signal exc_stop		: in std_logic;
			signal inh_or		: in std_logic;
			signal inh_stop		: in std_logic;

			-- output
			signal next_state	: out states) is
		begin

			if exc_or ='1'
			then
				from_exc_stop_on(

					-- input
					exc_stop	=> exc_stop,

					-- output
					next_state	=> next_state		
				);
			else
				
				from_inh_or_on(
			
					-- input
					inh_or		=> inh_or,
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state
				);
			end if;

		end procedure from_exc_or_on;


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
					next_state <= rest;
				else
					from_exc_or_on(
						-- input
						exc_or		=> exc_or,
						exc_stop	=> exc_stop,
						inh_or		=> inh_or,
						inh_stop	=> inh_stop,
                                                                           
						-- output          
						next_state	=> next_state
	
					);
				end if;

			-- exc_update
			when exc_update =>

				from_exc_stop_on(

					-- input
					exc_stop	=> exc_stop,

					-- output
					next_state	=> next_state		
				);		

			-- exc_end
			when exc_end =>		

				from_inh_or_on(
			
					-- input
					inh_or		=> inh_or,
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state
				);

			-- inh_update
			when inh_update =>

				from_inh_stop_on(

					-- input
					inh_stop	=> inh_stop,

					-- output
					next_state	=> next_state		
				);		

			-- inh_end
			when inh_end =>

				next_state <= sample_spikes;		

			-- rest
			when rest =>

				next_state <= idle;

			when others =>

				next_state <= reset;

		end case;

	end process state_evaluation;




	output_evaluation	: process(present_state)
	begin

		-- default values
		sample			<= '0';
		snn_ready		<= '0';
		exc_en			<= '0';
		anticipate_exc		<= '0';
		inh_en			<= '0';
		anticipate_inh		<= '0';
		exc_cnt_en		<= '0';
		exc_cnt_rst_n		<= '1';
		inh_cnt_en		<= '0';
		inh_cnt_rst_n		<= '1';
		exc_or_inh_sel		<= '0';
		mask1			<= '1';
		mask2			<= '1';
		cycles_cnt_en		<= '0';
		cycles_cnt_rst_n	<= '1';
		rest_en			<= '0';
		inh_elaboration		<= '0';


		case present_state is

			-- reset
			when reset =>

				exc_cnt_rst_n		<= '0';
				inh_cnt_rst_n		<= '0';
				cycles_cnt_rst_n	<= '0';
				mask1			<= '0';
				mask2			<= '0';

			-- idle
			when idle =>

				snn_ready		<= '1';
				mask1			<= '0';
				mask2			<= '0';

			-- sample_spikes
			when sample_spikes =>

				sample			<= '1';
				exc_en			<= '1';
				inh_en			<= '1';
				cycles_cnt_en		<= '1';
				anticipate_exc		<= '1';
				anticipate_inh		<= '1';

			-- exc_update
			when exc_update =>

				mask2			<= '0';
				exc_cnt_en		<= '1';

			-- exc_end
			when exc_end =>

				mask1			<= '0';
				mask2			<= '0';
				exc_cnt_rst_n		<= '0';

			-- inh_update
			when inh_update =>

				mask1			<= '0';
				inh_cnt_en		<= '1';
				inh_elaboration		<= '1';

			-- inh_end
			when inh_end =>

				mask1			<= '0';
				mask2			<= '0';
				inh_cnt_rst_n		<= '0';
				inh_elaboration		<= '1';

			-- rest
			when rest =>

				rest_en			<= '1';
				cycles_cnt_rst_n	<= '0';

			when others =>

				exc_cnt_rst_n		<= '0';
				inh_cnt_rst_n		<= '0';
				cycles_cnt_rst_n	<= '0';
				mask1			<= '0';
				mask2			<= '0';

		end case;

	end process output_evaluation;





	out_spikes	<= feedback_spikes;

	start1		<= exc_or_internal and mask1;
	start2		<= inh_or_internal and mask2;

	exc_or		<= exc_or_internal;
	inh_or		<= inh_or_internal;

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
			all_inputs		=> inh_or_internal,
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





end architecture test;
