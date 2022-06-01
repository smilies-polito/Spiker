library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity complete_neurons_tb is
end entity complete_neurons_tb;


architecture test of complete_neurons_tb is


	-- internal parallelism
	constant N		: integer := 16;
	constant N_weights	: integer := 15;

	-- number of neurons in the layer
	constant layer_size	: integer := 4;
	constant N_addr		: integer := 2;

	-- shift during the exponential decay
	constant shift		: integer := 1;

	-- model parameters
	constant v_th_value_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant inh_weight_int	: integer 	:= 7*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);

	-- control input
	signal clk		: std_logic;
	signal rst_n		: std_logic;		
	signal start		: std_logic;		
	signal stop		: std_logic;
	signal exc_or		: std_logic;
	signal exc_stop		: std_logic;
	signal inh_or		: std_logic;
	signal inh_stop		: std_logic;
	signal inh		: std_logic;
	signal load_v_th	: std_logic;
	signal neuron_addr	: std_logic_vector(N_addr-1 downto 0);

	-- input
	signal input_spike	: std_logic;

	-- input parameters
	signal v_th_value	: signed(N-1 downto 0);		
	signal v_reset		: signed(N-1 downto 0);		
	signal inh_weight	: signed(N-1 downto 0);		
	signal exc_weights	: signed(layer_size*N_weights-1 downto 0);

	-- output
	signal out_spikes	: std_logic_vector(layer_size-1 downto 0);
	signal all_ready	: std_logic;

	

	component complete_neurons is

		generic(

			-- internal parallelism
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
			clk		: in std_logic;
			rst_n		: in std_logic;		
			start		: in std_logic;		
			stop		: in std_logic;
			exc_or		: in std_logic;
			exc_stop	: in std_logic;
			inh_or		: in std_logic;
			inh_stop	: in std_logic;
			inh		: in std_logic;
			load_v_th	: in std_logic;
			neuron_addr	: in std_logic_vector(N_addr-1 downto 0);

			-- input
			input_spike	: in std_logic;

			-- input parameters
			v_th_value	: in signed(parallelism-1 downto 0);		
			v_reset		: in signed(parallelism-1 downto 0);		
			inh_weight	: in signed(parallelism-1 downto 0);		
			exc_weights	: in signed(N_neurons*
						weightsParallelism-1 downto 0);

			-- output
			out_spikes	: out std_logic_vector(N_neurons-1 downto 0);
			all_ready	: out std_logic
		);
		
	end component complete_neurons;

begin


	-- model parameters binary conversion
	v_th_value	<= to_signed(v_th_value_int, N);
	v_reset		<= to_signed(v_reset_int, N);
	inh_weight	<= to_signed(inh_weight_int, N);

	input_spike	<= '1';

	exc_weights_init	: process
	begin
		init	: for i in 0 to layer_size-1
		loop
			exc_weights((i+1)*N_weights-1 downto i*N_weights) <= 
					to_signed(exc_weight_int, N_weights);
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
	load_v_th_gen	: process
	begin

		load_v_th	<= '0';	-- 0ns
		wait for 26 ns;
		load_v_th	<= '1';	-- 26 ns
		wait for 48 ns;
		load_v_th	<= '0';	-- 74 ns
		wait;

	end process load_v_th_gen;


	-- neuron_addr
	neuron_addr_gen : process
	begin
		wait for 26 ns;
		
		for i in 0 to layer_size-1
		loop

			neuron_addr <= std_logic_vector(to_unsigned(i, N_addr));
			wait for 12 ns;

		end loop;

		wait for 156 ns;

		for i in 0 to layer_size-1
		loop

			neuron_addr <= std_logic_vector(to_unsigned(i, N_addr));
			wait for 12 ns;

		end loop;
		wait;
	end process neuron_addr_gen;




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


	-- exc_or
	exc_or_gen : process
	begin
		exc_or	<= '0';		-- 0 ns
		wait for 170 ns;			
		exc_or	<= '1';         -- 170  ns
		wait for 12 ns;			          
		exc_or	<= '0';         -- 182 ns
		wait;
	end process exc_or_gen;


	-- exc_stop
	exc_stop_gen : process
	begin
		exc_stop	<= '0';		-- 0 ns
		wait for 206 ns;			
		exc_stop	<= '1';         -- 206 ns
		wait for 12 ns;			          
		exc_stop	<= '0';         -- 218 ns
		wait;
	end process exc_stop_gen;


	-- inh
	inh_gen		: process
	begin
		inh	<= '0';
		wait for 230 ns;	
		inh	<= '1';		-- 242 ns
		wait for 36 ns;		          
		inh	<= '0';         -- 254 ns
		wait;
	end process inh_gen;


	-- inh_or
	inh_or_gen	: process
	begin
		inh_or	<= '0';		-- 0 ns
		wait for 218 ns;			          
		inh_or	<= '1';         -- 218 ns
		wait for 12 ns;			          
		inh_or	<= '0';         -- 230 ns
		wait;
	end process inh_or_gen;


	-- inh_stop
	inh_stop_gen	: process
	begin
		inh_stop	<= '0';		-- 0 ns
		wait for 254 ns;			
		inh_stop	<= '1';         -- 254 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 266 ns
		wait for 24 ns;			          
		inh_stop	<= '1';         -- 290 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 302 ns
		wait;
	end process inh_stop_gen;



	-- stop
	stop_gen : process
	begin
		stop	<= '0';		-- 0 ns
		wait for 386 ns;
		stop	<= '1';		-- 386 ns
		wait for 12 ns;
		stop <= '0';		-- 398 ns
		wait;
	end process stop_gen;



	dut	: complete_neurons

		generic map(
			-- internal parallelism
			parallelism		=> N,
			weightsParallelism	=> N_weights,

			-- number of neurons in the layer
			N_neurons	=> layer_size,
			N_addr		=> N_addr,

			-- shift during the exponential decay
			shift		=> shift
		)

		port map(
			-- control input
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,
			exc_or		=> exc_or,
			exc_stop	=> exc_stop,
			inh_or		=> inh_or,
			inh_stop	=> inh_stop,
			inh		=> inh,
			load_v_th	=> load_v_th,
			neuron_addr	=> neuron_addr,

			-- input
			input_spike	=> input_spike,

			-- input parameters
			v_th_value	=> v_th_value,
			v_reset		=> v_reset,
			inh_weight	=> inh_weight,
			exc_weights	=> exc_weights,

			-- output
			out_spikes	=> out_spikes,
			all_ready	=> all_ready
		);

end architecture test;
