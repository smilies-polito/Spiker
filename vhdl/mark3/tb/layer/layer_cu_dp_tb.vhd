library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity layer_cu_dp_tb is
end entity layer_cu_dp_tb;

architecture test of layer_cu_dp_tb is


	-- int parallelism
	constant parallelism		: integer := 32;
	constant weightsParallelism	: integer := 31;

	-- input spikes
	constant N_inputs		: integer := 2;

	-- must be one bit larger that the parallelism required to count
	-- up to N_inputs
	constant N_inputs_cnt		: integer := 2;

	-- inhibitory spikes
	constant N_neurons		: integer := 3;

	-- must be one bit larger that the parallelism required to count
	-- up to N_neurons
	constant N_neurons_cnt		: integer := 3;

	-- exponential decay shift
	constant shift			: integer := 10;

	-- model parameters
	constant v_th_value_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^11 rounded	
	constant inh_weight_int	: integer 	:= 5*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);

	-- control input
	signal clk			: std_logic;
	signal rst_n			: std_logic;
	signal start			: std_logic;
	signal stop			: std_logic;		
	signal init_v_th		: std_logic;

	-- address to select the neurons
	signal v_th_addr		: std_logic_vector(N_neurons_cnt-1
						downto 0);

	-- data input
	signal input_spikes		: std_logic_vector(N_inputs-1 downto 0);


	-- input parameters
	signal v_th_value		: signed(parallelism-1 downto 0);		
	signal v_reset			: signed(parallelism-1 downto 0);	
	signal inh_weight		: signed(parallelism-1 downto 0);		
	signal exc_weights		: signed
						(N_neurons*weightsParallelism-1 
						 downto 0);

	-- terminal counters 
	signal N_inputs_tc		: std_logic_vector
					     (N_inputs_cnt-1 downto 0);
	signal N_neurons_tc		: std_logic_vector
						(N_neurons_cnt-1 downto 0);

	-- from datapath towards control unit
	signal exc_or			: std_logic;		
	signal exc_stop			: std_logic;		
	signal inh_or			: std_logic;		
	signal inh_stop			: std_logic;

	-- from control unit towards datapath
	signal exc_en			: std_logic;	
	signal anticipate_exc		: std_logic;	
	signal inh_en			: std_logic;	
	signal anticipate_inh		: std_logic;	
	signal exc_cnt_en		: std_logic;	
	signal exc_cnt_rst_n		: std_logic;	
	signal inh_cnt_en		: std_logic;	
	signal inh_cnt_rst_n		: std_logic;	
	signal exc_or_inh_sel		: std_logic;	
	signal inh			: std_logic;	

	-- output
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;	
	signal sample			: std_logic;
	signal layer_ready		: std_logic;
	signal all_ready		: std_logic;
	signal out_spikes		: std_logic_vector(N_neurons-1 downto 0);

	-- output address to select the excitatory weights
	signal exc_cnt			: std_logic_vector
						(N_inputs_cnt-1 downto 0);


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
	signal exc_spikes		: std_logic_vector(N_inputs-1 downto 0);
	signal inh_spikes		: std_logic_vector(N_neurons-1 downto 0);
	signal exc_spike		: std_logic;
	signal inh_spike		: std_logic;
	signal inh_cnt			: std_logic_vector(N_neurons_cnt-1 downto 0);
	signal spike			: std_logic;
	signal exc_or_int		: std_logic;
	signal exc_stop_int		: std_logic;
	signal inh_or_int		: std_logic;
	signal inh_stop_int		: std_logic;
	signal feedback_spikes		: std_logic_vector(N_neurons-1 downto 0);
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



	component complete_neurons is

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
		
	end component complete_neurons;


begin

	out_spikes	<= feedback_spikes;
	exc_or		<= exc_or_int;
	exc_stop	<= exc_stop_int;
	inh_or		<= inh_or_int;
	inh_stop	<= inh_stop_int;


	N_neurons_tc	<= "011";
	N_inputs_tc	<= "10";


	-- model parameters binary conversion
	v_th_value	<= to_signed(v_th_value_int, parallelism);
	v_reset		<= to_signed(v_reset_int, parallelism);
	inh_weight	<= to_signed(inh_weight_int, parallelism);

	exc_weights_init	: process
	begin
		init	: for i in 0 to N_neurons-1
		loop
			exc_weights((i+1)*weightsParallelism-1 downto 
					i*weightsParallelism) <= 
					to_signed(exc_weight_int, 
					weightsParallelism);
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


	-- load_v_th
	init_v_th_gen	: process
	begin

		init_v_th	<= '0';	-- 0ns
		wait for 26 ns;
		init_v_th	<= '1';	-- 26 ns
		wait for 48 ns;
		init_v_th	<= '0';	-- 74 ns
		wait;

	end process init_v_th_gen;


	-- v_th_addr
	v_th_addr_gen : process
	begin
		wait for 26 ns;
		
		for i in 0 to N_neurons-1
		loop

			v_th_addr <= std_logic_vector(to_unsigned(i,
						N_neurons_cnt));
			wait for 12 ns;

		end loop;

		-- wait for 156 ns;

		-- v_th_addr	<= "11";	-- 230 ns

		wait;
	end process v_th_addr_gen;




	-- start
	start_gen : process
	begin
		start	<= '0';		-- 0 ns
		wait for 158 ns;
		start	<= '1';		-- 158 ns
		wait for 12 ns;
		start	<= '0';		-- 170 ns
		wait;
	end process start_gen;



	-- input_spikes
	input_spikes_gen: process
	begin
		input_spikes	<= "00"; -- 0 ns	
		wait for 182 ns;
		input_spikes	<= "11"; -- 182 ns
		wait for 48 ns;
		input_spikes	<= "00"; -- 230 ns
		wait for 48 ns;
		input_spikes	<= "01"; -- 258 ns
		wait for 84 ns;
		input_spikes	<= "00"; -- 342 ns
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



	-- output evaluation
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



	bare_layer : complete_neurons

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
			all_ready		=> all_ready
		);



end architecture test;
