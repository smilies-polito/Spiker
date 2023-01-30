library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity bare_neurons_tb is
end entity bare_neurons_tb;



architecture test of bare_neurons_tb is

	-- parallelism
	constant neuron_bit_width		: integer := 16;
	constant weights_bit_width	: integer := 15;

	-- number of neurons in the layer
	constant N_neurons	: integer := 4;

	-- exponential shift
	constant shift		: integer := 10;

	-- model parameters
	constant v_th_value_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant inh_weight_int	: integer 	:= 7*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);


	-- input parameters
	signal v_th_value	: signed(neuron_bit_width-1 downto 0);
	signal v_reset		: signed(neuron_bit_width-1 downto 0);
	signal inh_weight	: signed(neuron_bit_width-1 downto 0);
	signal exc_weights	: signed(N_neurons*weights_bit_width-1 downto 0);

	-- control input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal stop	        : std_logic;
	signal exc_or	        : std_logic;
	signal exc_stop		: std_logic;
	signal inh_or		: std_logic;
	signal inh_stop		: std_logic;
	
	-- to load the threshold
	signal v_th_en		: std_logic_vector(N_neurons-1 downto 0);

	-- input
        signal input_spikes	: std_logic_vector(N_neurons-1 downto 0);


	-- output
	signal out_spikes	: std_logic_vector(N_neurons-1 downto 0);
	signal all_ready	: std_logic;



	component bare_neurons is

		generic(
			-- internal parallelism
			neuron_bit_width	: integer := 16;
			weights_bit_width	: integer := 5;

			-- number of neurons in the layer
			N_neurons	: integer := 400;

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
			v_th_en		: in std_logic_vector(N_neurons-1 downto 0);

			-- input
			input_spikes	: in std_logic_vector(N_neurons-1 downto 0);

			-- input parameters
			v_th_value	: in signed(neuron_bit_width-1 downto 0);		
			v_reset		: in signed(neuron_bit_width-1 downto 0);		
			inh_weight	: in signed(neuron_bit_width-1 downto 0);		
			exc_weights	: in signed(N_neurons*weights_bit_width-1 downto 0);

			-- output
			out_spikes	: out std_logic_vector(N_neurons-1 downto 0);
			all_ready	: out std_logic
		);

	end component bare_neurons;



begin


	-- model parameters binary conversion
	v_th_value	<= to_signed(v_th_value_int, neuron_bit_width);
	v_reset		<= to_signed(v_reset_int, neuron_bit_width);
	inh_weight	<= to_signed(inh_weight_int, neuron_bit_width);

	exc_weights_init	: process
	begin
		init	: for i in 0 to N_neurons-1
		loop
			exc_weights((i+1)*weights_bit_width-1 downto i*weights_bit_width) <= 
					to_signed(exc_weight_int, weights_bit_width);
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



	-- v_th_en
	v_th_en_gen : process
	begin
		v_th_en	<= (others => '0');		-- 0 ns
		wait for 26 ns;
		v_th_en	<= (others => '1');		-- 26 ns
		wait for 12 ns;
		v_th_en	<= (others => '0');		-- 38 ns
		wait;
	end process v_th_en_gen;




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


	-- exc_or
	exc_or_gen : process
	begin
		exc_or	<= '0';		-- 0 ns
		wait for 62 ns;			
		exc_or	<= '1';         -- 62 ns
		wait for 48 ns;			          
		exc_or	<= '0';         -- 110 ns
		wait for 48 ns;			          
		exc_or	<= '1';         -- 158 ns
		wait for 84 ns;			          
		exc_or	<= '0';         -- 242 ns
		wait;
	end process exc_or_gen;


	-- exc_stop
	exc_stop_gen : process
	begin
		exc_stop	<= '0';		-- 0 ns
		wait for 98 ns;			
		exc_stop	<= '1';         -- 98 ns
		wait for 12 ns;			          
		exc_stop	<= '0';         -- 110 ns
		wait for 84 ns;			          
		exc_stop	<= '1';         -- 194 ns
		wait for 12 ns;			          
		exc_stop	<= '0';         -- 206 ns
		wait;
	end process exc_stop_gen;


	-- inh_or
	inh_or_gen : process
	begin
		inh_or	<= '0';		-- 0 ns
		wait for 110 ns;			          
		inh_or	<= '1';         -- 110 ns
		wait for 132 ns;			          
		inh_or	<= '0';         -- 242 ns
		wait;
	end process inh_or_gen;


	-- inh_stop
	inh_stop_gen : process
	begin
		inh_stop	<= '0';		-- 0 ns
		wait for 146 ns;			
		inh_stop	<= '1';         -- 146 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 158 ns
		wait for 72 ns;			          
		inh_stop	<= '1';         -- 230 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 242 ns
		wait for 12 ns;
		inh_stop	<= '1';		-- 254 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 266 ns
		wait;
	end process inh_stop_gen;


	-- input_spikes
	input_spikes_gen: process
	begin
		input_spikes	<= (others => '0');	-- 0 ns	
		wait for 62 ns;	          
		input_spikes	<= (others => '1');	-- 62 ns
		wait for 96 ns;					  
		input_spikes	<= (others => '0');	-- 158 ns
		wait for 24 ns;				      
		input_spikes	<= (others => '1');	-- 182 ns
		wait for 12 ns;				      
		input_spikes	<= (others => '0');	-- 194 ns
		wait for 12 ns;				      
		input_spikes	<= (others => '1');	-- 206 ns
		wait for 12 ns;				      
		input_spikes	<= (others => '0');	-- 218 ns
		wait for 12 ns;				      
		input_spikes	<= (others => '1');	-- 230 ns
		wait for 12 ns;				      
		input_spikes	<= (others => '0');	-- 242 ns
		wait;
	end process input_spikes_gen;




	-- stop
	stop_gen : process
	begin
		stop	<= '0';		-- 0 ns
		wait for 266 ns;
		stop	<= '1';		-- 266 ns
		wait for 12 ns;
		stop <= '0';		-- 278 ns
		wait;
	end process stop_gen;




	dut : bare_neurons

		generic map(
			-- parallelism
			neuron_bit_width	=> neuron_bit_width,	
			weights_bit_width	=> weights_bit_width,

			-- number of neurons in the layer
			N_neurons	=> N_neurons,

			-- shift amount
			shift		=> shift
		)

		port map(
			-- input controls
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,
			exc_or	       	=> exc_or,
			exc_stop       	=> exc_stop,
			inh_or	        => inh_or,
			inh_stop        => inh_stop,

			-- to load the threshold
			v_th_en		=> v_th_en,

			-- input
                       	input_spikes	=> input_spikes,

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
