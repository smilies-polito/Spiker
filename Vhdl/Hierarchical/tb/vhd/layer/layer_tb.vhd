library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity layer_tb is
end entity layer_tb;


architecture test of layer_tb is


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
	constant inh_weight_int	: integer 	:= 3*(2**10);	
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

	-- output
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;	
	signal sample			: std_logic;
	signal ready			: std_logic;
	signal out_spikes		: std_logic_vector(N_neurons-1 downto 0);

	-- output address to select the excitatory weights
	signal exc_cnt			: std_logic_vector
						(N_inputs_cnt-1 downto 0);



	component layer is

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
			rst_n			: in std_logic;	
			start			: in std_logic;	
			stop			: in std_logic;	
			init_v_th		: in std_logic;

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

			-- output
			out_spikes		: out std_logic_vector
							(N_neurons-1 downto 0);
			ready			: out std_logic;
			sample			: out std_logic;
			cycles_cnt_rst_n	: out std_logic;	
			cycles_cnt_en		: out std_logic;	

			-- output address to select the excitatory weights
			exc_cnt			: out std_logic_vector
							(N_inputs_cnt-1 downto 0)
		);

	end component layer;



begin



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



	dut	: layer

		generic map(

			-- internal parallelism
			parallelism		=> parallelism,	
			weightsParallelism	=> weightsParallelism,
                                                                   
			-- input spikes       
			N_inputs		=> N_inputs,

			-- must be one bit larger that the parallelism required
			-- to count up to N_inputs
			N_inputs_cnt		=> N_inputs_cnt,
                                                                   
			-- inhibitory spikes       
			N_neurons		=> N_neurons,

			-- must be one bit larger that the parallelism required
			-- to count up to N_neurons
			N_neurons_cnt		=> N_neurons_cnt,
                                                                   
			-- exponential decay shift 
			shift			=> shift
				
		)

		port map(
			-- input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			stop			=> stop,
			init_v_th		=> init_v_th,

			-- address to select the neurons
			v_th_addr		=> v_th_addr,

			-- data input
			input_spikes		=> input_spikes,
                                                                   
			-- input parameters        
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			exc_weights		=> exc_weights,
						   		
                                                                   
			-- number of inputs neuron
			N_inputs_tc		=> N_inputs_tc,
			N_neurons_tc		=> N_neurons_tc,
                                                                   
			-- output                  
			out_spikes		=> out_spikes,
			sample			=> sample,
			ready			=> ready,
			cycles_cnt_rst_n	=> cycles_cnt_rst_n,	
			cycles_cnt_en		=> cycles_cnt_en,

			-- output address to select the excitatory weights
			exc_cnt			=> exc_cnt
		);







end architecture test;
