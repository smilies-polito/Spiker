library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;


entity debug_neuron_tb is
end entity debug_neuron_tb;


architecture behaviour of debug_neuron_tb is


	-- parallelism
	constant N		: integer := 16;
	constant N_weight	: integer := 15;

	-- exponential shift
	constant shift		: integer := 10;

	-- model parameters
	constant v_th_value_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant inh_weight_int	: integer 	:= -7*(2**10);	
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

	-- debug output
	signal v_out		: signed(N-1 downto 0);

	-- file signals
	signal w_en		: std_logic;



	component debug_neuron is

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
			neuron_ready	: out std_logic;

			-- debug output
			v_out		: out signed(N-1 downto 0)
		);

	end component debug_neuron;




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
		wait for 12 ns;			
		exc_or	<= '0';         -- 74 ns
		wait;
	end process exc_or_gen;


	-- exc_stop
	exc_stop_gen : process
	begin
		exc_stop	<= '0';		-- 0 ns
		wait for 98 ns;
		exc_stop	<= '1';		-- 98 ns
		wait for 12 ns;
		exc_stop	<= '0';		-- 110 ns
		wait;
	end process exc_stop_gen;


	-- inh_or
	inh_or_gen : process
	begin
		inh_or	<= '0';		-- 0 ns
		wait for 10000 ns;
		inh_or	<= '1';		-- 10000 ns
		wait for 12 ns;
		inh_or	<= '0';		-- 10012 ns
		wait;
	end process inh_or_gen;


	-- inh_stop
	inh_stop_gen : process
	begin
		inh_stop	<= '0';		-- 0 ns
		wait for 122 ns;
		inh_stop	<= '1';		-- 122 ns
		wait for 12 ns;
		inh_stop	<= '0';		-- 134 ns
		wait for 9890 ns;
		inh_stop	<= '1';		-- 10024 ns
		wait for 24 ns;
		inh_stop	<= '0';		-- 10048 ns
		wait;
	end process inh_stop_gen;


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';	-- 0 ns	
		wait for 74 ns;
		input_spike	<= '1';	-- 74 ns	
		wait for 12 ns;
		input_spike	<= '0';	-- 86 ns	
		wait for 9926 ns;
		input_spike	<= '1';	-- 10012 ns	
		wait for 12 ns;
		input_spike	<= '0';	-- 10024 ns	
		wait;
	end process input_spike_gen;




	-- stop
	stop_gen : process
	begin
		stop	<= '0';		-- 0 ns
		wait;
	end process stop_gen;




	dut : debug_neuron 

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
			neuron_ready	=> neuron_ready,

			-- debug output
			v_out		=> v_out
		);



	write_enable : process
	begin
		w_en <= '0';
		wait for 26 ns;
		w_en <= '1';
		wait;

	end process write_enable;




	file_write : process(clk, w_en)

		file in_spikes_file	: text open write_mode is
			"/home/alessio/Documents/Poli/Dottorato/progetti/snn/spiker/vhdl/mark3/plot/data/inSpikes.txt";

		file out_spikes_file	: text open write_mode is
			"/home/alessio/Documents/Poli/Dottorato/progetti/snn/spiker/vhdl/mark3/plot/data/outSpikes.txt";

		file v_file		: text open write_mode is 
			"/home/alessio/Documents/Poli/Dottorato/progetti/snn/spiker/vhdl/mark3/plot/data/v.txt";

		variable row		: line;
		variable write_var	: integer;

	begin

		if clk'event and clk = '1'
		then

			if w_en = '1'
			then

				-- write the input spike
				write(row, input_spike, right, 1);
				writeline(in_spikes_file, row);

				-- write the potential value
				write_var := to_integer(v_out);
				write(row, write_var);
				writeline(v_file, row);

				-- write the output spike
				write(row, out_spike, right, 1);
				writeline(out_spikes_file, row);

			end if;

		end if;

	end process file_write;




end architecture behaviour;
