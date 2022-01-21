library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_tb is
end entity neuron_tb;



architecture behaviour of neuron_tb is


	-- parallelism
	constant N		: integer := 16;
	constant N_weight	: integer := 15;

	-- exponential shift
	constant shift		: integer := 10;

	-- model parameters
	constant v_th_value_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant inh_weight_int	: integer 	:= 7*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);


	-- input parameters
	signal v_th_value	: signed(N-1 downto 0);
	signal v_reset		: signed(N-1 downto 0);
	signal inh_weight	: signed(N-1 downto 0);
	signal exc_weight	: signed(N_weight-1 downto 0);


	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal stop	        : std_logic;
	signal exc_or	        : std_logic;
	signal exc_stop		: std_logic;
	signal inh_or		: std_logic;
	signal inh_stop		: std_logic;
        signal input_spike	: std_logic;

	-- to load the threshold
	signal v_th_en		: std_logic;

	-- output
	signal out_spike	: std_logic;
	signal neuron_ready	: std_logic;





	component neuron is

		generic(
			-- parallelism
			N		: integer := 16;
			N_weight	: integer := 5;

			-- shift amount
			shift		: integer := 1
		);

		port(
			-- input controls
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;
			exc_or		: in std_logic;
			exc_stop	: in std_logic;
			inh_or		: in std_logic;
			inh_stop	: in std_logic;
			input_spike	: in std_logic;

			-- to load the threshold
			v_th_en		: in std_logic;

			-- input parameters
			v_th_value	: in signed(N-1 downto 0);
			v_reset		: in signed(N-1 downto 0);
			inh_weight	: in signed(N-1 downto 0);
			exc_weight	: in signed(N_weight-1 downto 0);

			-- output
			out_spike	: out std_logic;
			neuron_ready	: out std_logic
		);

	end component neuron;




begin



	-- model parameters binary conversion
	v_th_value	<= to_signed(v_th_value_int, N);
	v_reset		<= to_signed(v_reset_int, N);
	inh_weight	<= to_signed(inh_weight_int, N);
	exc_weight	<= to_signed(exc_weight_int, N_weight);



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
		v_th_en	<= '0';		-- 0 ns
		wait for 26 ns;
		v_th_en	<= '1';		-- 26 ns
		wait for 12 ns;
		v_th_en	<= '0';		-- 38 ns
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


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';	-- 0 ns	
		wait for 62 ns;	          
		input_spike	<= '1'; -- 62 ns
		wait for 96 ns;			          
		input_spike	<= '0'; -- 158 ns
		wait for 24 ns;			          
		input_spike	<= '1'; -- 182 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 194 ns
		wait for 12 ns;			          
		input_spike	<= '1'; -- 206 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 218 ns
		wait for 12 ns;			          
		input_spike	<= '1'; -- 230 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 242 ns
		wait;
	end process input_spike_gen;




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




	dut : neuron 

		generic map(
			-- parallelism
			N		=> N,	
			N_weight	=> N_weight,

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
                       	input_spike	=> input_spike,

			-- to load the threshold
			v_th_en		=> v_th_en,

			-- input parameters
			v_th_value	=> v_th_value,
			v_reset		=> v_reset,
			inh_weight	=> inh_weight,
			exc_weight	=> exc_weight,
                                                       
			-- output          
			out_spike	=> out_spike,
			neuron_ready	=> neuron_ready
		);


end architecture behaviour;
