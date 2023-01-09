library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neurons_layer_tb is
end entity neurons_layer_tb;



architecture behaviour of neurons_layer_tb is

	-- parallelism
	constant N		: integer := 16;

	-- number of neurons in the layer
	constant layer_size	: integer := 4;

	-- exponential shift
	constant shift		: integer := 10;

	-- model parameters
	constant v_th_0_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^11 rounded	
	constant inh_weight_int	: integer 	:= 7*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);


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
	signal stop	        : std_logic;
	signal exc_or	        : std_logic;
	signal exc_stop		: std_logic;
	signal inh_or		: std_logic;
	signal inh_stop		: std_logic;
        signal input_spikes	: std_logic_vector(layer_size-1 downto 0);


	-- output
	signal out_spikes	: std_logic_vector(layer_size-1 downto 0);
	signal neurons_ready	: std_logic;






	component neurons_layer is

		generic(
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
			input_spikes	: in std_logic_vector(layer_size-1
						  downto 0);

			-- input parameters
			v_th_0		: in signed(N-1 downto 0);		
			v_reset		: in signed(N-1 downto 0);		
			inh_weight	: in signed(N-1 downto 0);		
			v_th_plus	: in signed(N-1 downto 0);		
			exc_weights	: in signed(layer_size*N-1 downto 0);

			-- output
			out_spikes	: out std_logic_vector(layer_size-1 
						downto 0);
			neurons_ready	: out std_logic
		);

	end component neurons_layer;

begin


	-- model parameters binary conversion
	v_th_0		<= to_signed(v_th_0_int, N);
	v_reset		<= to_signed(v_reset_int, N);
	v_th_plus	<= to_signed(v_th_plus_int, N);
	inh_weight	<= to_signed(inh_weight_int, N);

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
		input_spikes	<= (others => '1'); -- 62 ns
		wait for 96 ns;
		input_spikes	<= (others => '0'); -- 158 ns
		wait for 24 ns;
		input_spikes	<= (others => '1'); -- 182 ns
		wait for 12 ns;
		input_spikes	<= (others => '0'); -- 194 ns
		wait for 12 ns;
		input_spikes	<= (others => '1'); -- 206 ns
		wait for 12 ns;
		input_spikes	<= (others => '0'); -- 218 ns
		wait for 12 ns;
		input_spikes	<= (others => '1'); -- 230 ns
		wait for 12 ns;
		input_spikes	<= (others => '0'); -- 242 ns
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







	dut : neurons_layer

		generic map(
			-- parallelism
			N		=> N,	

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
			stop		=> stop,
			exc_or	       	=> exc_or,
			exc_stop       	=> exc_stop,
			inh_or	        => inh_or,
			inh_stop        => inh_stop,
                       	input_spikes	=> input_spikes,

			-- input parameters
			v_th_0		=> v_th_0,
			v_reset		=> v_reset,
			inh_weight	=> inh_weight,
			exc_weights	=> exc_weights,
			v_th_plus	=> v_th_plus,
                                                       
			-- output          
			out_spikes	=> out_spikes,
			neurons_ready	=> neurons_ready
		);


end architecture behaviour;
