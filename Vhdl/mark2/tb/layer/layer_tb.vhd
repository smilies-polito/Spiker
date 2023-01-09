library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity layer_tb is
end entity layer_tb;


architecture test of layer_tb is

	-- internal parallelism
	constant parallelism		: integer := 32;

	-- excitatory spikes
	constant input_parallelism	: integer := 784;
	constant N_exc_cnt		: integer := 11;

	-- inhibitory spikes
	constant layer_size		: integer := 400;
	constant N_inh_cnt		: integer := 10;

	-- elaboration steps
	constant N_cycles_cnt		: integer := 12;

	-- exponential decay shift
	constant shift			: integer := 10;

	-- model parameters
	constant v_th_0_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^11 rounded	
	constant inh_weight_int	: integer 	:= -1;	
	constant exc_weight_int	: integer 	:= 17;


	-- number of inputs, neurons and cycles
	signal N_inputs		: std_logic_vector
					(N_exc_cnt-1 downto 0);
	signal N_neurons	: std_logic_vector
					(N_inh_cnt-1 downto 0);
	signal N_cycles		: std_logic_vector
					(N_cycles_cnt-1 downto 0);
	-- input parameters
	signal v_th_0		: signed(parallelism-1 downto 0);
	signal v_reset		: signed(parallelism-1 downto 0);
	signal v_th_plus	: signed(parallelism-1 downto 0);
	signal inh_weight	: signed(parallelism-1 downto 0);
	signal exc_weights	: signed(layer_size*parallelism-1 downto 0);

	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal input_spikes	: std_logic_vector
					(input_parallelism-1 downto 0);

	-- output
	signal sample		: std_logic;
	signal ready		: std_logic;
	signal out_spikes	: std_logic_vector(layer_size-1 downto 0);
	signal exc_cnt		: std_logic_vector(N_exc_cnt-1 downto 0);




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




begin


	N_neurons	<= std_logic_vector(to_unsigned(400, N_inh_cnt));
	N_inputs	<= std_logic_vector(to_unsigned(784, N_exc_cnt));
	N_cycles	<= std_logic_vector(to_unsigned(3500, N_cycles_cnt));


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
		input_spikes	<= (others => '0'); 			-- 0 ns	
		wait for 62 ns;
		input_spikes	<= (others => '1'); 			-- 62 ns
		wait for 48 ns;
		input_spikes	<= (others => '0'); 			-- 110 ns
		wait for 48 ns;
		input_spikes	<= std_logic_vector(to_unsigned(1,
					input_parallelism)); 		-- 158 ns
		wait for 84 ns;
		input_spikes	<= (others => '0'); 			-- 242 ns
		wait;
	end process input_spikes_gen;






	dut	: layer

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
						   		
                                                                   
			-- number of inputs neuron
			N_inputs		=> N_inputs,
			N_neurons		=> N_neurons,
			N_cycles		=> N_cycles,
                                                                   
			-- output                  
			exc_cnt			=> exc_cnt,
			out_spikes		=> out_spikes,
			sample			=> sample,
			ready			=> ready			
		);







end architecture test;
