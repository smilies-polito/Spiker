library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neurons_layer_tb is
end entity neurons_layer_tb;



architecture behaviour of neurons_layer_tb is


	-- neurons counter parallelism
	constant N_cnt		: integer := 2;

	-- internal parallelism
	constant N		: integer := 8;

	-- number of neurons in the layer
	constant layer_size	: integer := 3;

	-- shift during the exponential decay
	constant shift		: integer := 1;

	-- model parameters
	constant v_th_0_int	: integer := 13;	
	constant v_reset_int	: integer := 5;	
	constant v_th_plus_int	: integer := 1;	
	constant inh_weight_int	: integer := -15;	
	constant exc_weight_int	: integer := 3;


	-- input parameters
	signal v_th_0		: signed(N-1 downto 0);
	signal v_reset		: signed(N-1 downto 0);
	signal v_th_plus	: signed(N-1 downto 0);
	signal inh_weight	: signed(N-1 downto 0);
	signal exc_weights	: signed(layer_size*N-1 downto 0);


	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal start1		: std_logic;
	signal start2		: std_logic;
	signal rest_en		: std_logic;
	signal neuron_cnt	: std_logic_vector(N_cnt-1 downto 0);
	signal input_spike	: std_logic;

	-- output
	signal out_spikes	: std_logic_vector(layer_size-1 downto 0);
	signal layer_ready	: std_logic;




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

			-- input parameters
			v_th_0		: in signed(N-1 downto 0);		
			v_reset		: in signed(N-1 downto 0);		
			inh_weight	: in signed(N-1 downto 0);		
			v_th_plus	: in signed(N-1 downto 0);		
			exc_weights	: in signed(layer_size*N-1 downto 0);

			-- output
			out_spikes	: out 
				std_logic_vector(layer_size-1 downto 0);
			layer_ready	: out std_logic
		);

	end component neurons_layer;



begin


	-- model parameters binary conversion
	v_th_0				<= to_signed(v_th_0_int, N);
	v_reset				<= to_signed(v_reset_int, N);
	v_th_plus			<= to_signed(v_th_plus_int, N);
	inh_weight			<= to_signed(inh_weight_int, N);

	exc_weights_init	: process
	begin
		init	: for i in 0 to layer_size-1
		loop
			exc_weights((i+1)*N-1 downto i*N) <= 
					to_signed(exc_weight_int, N);
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
		wait for 26 ns;
		start	<= '1';		-- 26 ns
		wait for 12 ns;
		start	<= '0';		-- 38 ns
		wait for 206 ns;
		start	<= '1';		-- 254 ns
		wait for 12 ns;
		start	<= '0';		-- 266 ns
		wait for 144 ns;
		start	<= '1';		-- 410 ns
		wait for 12 ns;
		start	<= '0';		-- 422 ns
		wait;
	end process start_gen;


	-- start1
	start1_gen : process
	begin
		start1	<= '0';		-- 0 ns
		wait for 38 ns;			
		start1	<= '1';		-- 38 ns
		wait for 24 ns;			          
		start1	<= '0';         -- 62 ns
		wait for 48 ns;			          
		start1	<= '1';         -- 110 ns
		wait for 12 ns;			          
		start1	<= '0';         -- 122 ns
		wait for 36 ns;
		start1	<= '1';         -- 158 ns
		wait for 12 ns;
		start1	<= '0';         -- 170 ns
		wait for 96 ns;
		start1	<= '1';		-- 266 ns
		wait for 60 ns;
		start1	<= '0';		-- 326 ns
		wait for 96 ns;
		start1 <= '1';		-- 422 ns
		wait for 60 ns;
		start1 <= '0';		-- 482 ns
		wait;
	end process start1_gen;



	-- start2
	start2_gen : process
	begin
		start2	<= '0';		-- 0 ns
		wait for 62 ns;		
		start2	<= '1';		-- 62 ns
		wait for 24 ns;			          
		start2	<= '0';         -- 86 ns
		wait for 48 ns;		
		start2	<= '1';		-- 134 ns
		wait for 12 ns;
		start2	<= '0';		-- 146 ns
		wait for 24 ns;
		start2	<= '1';		-- 170 ns
		wait for 12 ns;
		start2	<= '0';		-- 182 ns
		wait;
	end process start2_gen;


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';	-- 0 ns	
		wait for 38 ns;			
		input_spike	<= '1';	-- 38 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 50 ns
		wait for 12 ns;			          
		input_spike	<= '1'; -- 62 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 74 ns
		wait for 36 ns;			          
		input_spike	<= '1'; -- 110 ns
		wait for 48 ns;			          
		input_spike	<= '0'; -- 158 ns
		wait for 108 ns;
		input_spike	<= '1'; -- 266 ns
		wait for 60 ns;
		input_spike	<= '0'; -- 326 ns
		wait for 96 ns;
		input_spike	<= '1'; -- 422 ns
		wait for 48 ns;
		input_spike	<= '0';	-- 470 ns
		wait;
	end process input_spike_gen;


	-- neuron_cnt
	neuron_cnt_gen : process
	begin
		neuron_cnt	<= (others => '0');	-- 0 ns
		wait;
	end process neuron_cnt_gen;



	-- rest_en
	rest_en_gen : process
	begin
		rest_en	<= '0';		-- 0 ns
		wait for 206 ns;
		rest_en	<= '1';		-- 206 ns
		wait for 12 ns;
		rest_en <= '0';		-- 218 ns
		wait for 156 ns;
		rest_en <= '1';		-- 374 ns
		wait for 12 ns;
		rest_en <= '0';		-- 386 ns
		wait;
	end process rest_en_gen;




	dut	: neurons_layer 
		generic map(
			-- neurons counter parallelism
			N_cnt		=> N_cnt,

			-- internal parallelism
			N		=> N,

			-- number of neurons in the layer
			layer_size	=> layer_size,

			-- shift during the exponential decay
			shift		=> shift
		)

		port map(
			-- control input
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			start1		=> start1,
			start2		=> start2,
			rest_en		=> rest_en,
			input_spike	=> input_spike,
			neuron_cnt	=> neuron_cnt,

			-- input parameters
			v_th_0		=> v_th_0,
			v_reset		=> v_reset,
			inh_weight	=> inh_weight,
			v_th_plus	=> v_th_plus,
			exc_weights	=> exc_weights,
                                                      
			-- output          -- output
			out_spikes	=> out_spikes,
			layer_ready	=> layer_ready
		);


end architecture behaviour;
