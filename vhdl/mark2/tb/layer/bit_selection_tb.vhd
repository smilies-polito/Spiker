library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bit_selection_tb is
end entity bit_selection_tb;



architecture behaviour of bit_selection_tb is


	-- number of input bits
	constant N_bit			: integer := 4;

	-- selction counter parallelism
	constant N_cnt			: integer := 3;		

	-- input
	signal clk			: std_logic;
	signal input_bits		: std_logic_vector(N_bit-1 downto 0);
	signal select_cnt_en		: std_logic;
	signal select_cnt_rst_n		: std_logic;
	signal N_inputs			: std_logic_vector(N_cnt-1 downto 0);

	-- output
	signal all_inputs		:  std_logic;
	signal selected_input		:  std_logic;
	signal input_index		:  std_logic_vector(N_cnt-1 downto 0);
	signal stop			:  std_logic;		


	component bit_selection is

		generic(
			-- number of input bits
			N_bit			: integer := 8;

			-- selction counter parallelism
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


begin

	
	N_inputs	<= "100";

	-- clock
	clock_gen : process
	begin
		clk	<= '0';				-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         		-- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- reset
	cnt_rst_n_gen : process
	begin
		select_cnt_rst_n	<= '1';		-- 0 ns
		wait for 14 ns;
		select_cnt_rst_n	<= '0';		-- 14 ns
		wait for 12 ns;
		select_cnt_rst_n	<= '1';		-- 26 ns
		wait;
	end process cnt_rst_n_gen;


	-- enable
	cnt_en_gen : process
	begin
		select_cnt_en	<= '0';			-- 0 ns
		wait for 26 ns;
		select_cnt_en	<= '1';			-- 26 ns
		wait;
	end process cnt_en_gen;


	simulation	: process
	begin

		wait for 26 ns;

		for i in 0 to 2**N_bit
		loop

			input_bits	<= std_logic_vector(to_unsigned(i, N_bit));
			wait for 48 ns;

		end loop;

		wait;

	end process simulation;

	dut:	bit_selection

		generic map(
			-- number of input bits
			N_bit			=> N_bit,

			-- selction counter parallelism
			N_cnt			=> N_cnt
		)

		port map(
			-- input
			clk			=> clk,
			input_bits		=> input_bits,
			select_cnt_en		=> select_cnt_en,
			select_cnt_rst_n	=> select_cnt_rst_n,
			N_inputs		=> N_inputs,
                                                                   
			-- output                  -- output
			all_inputs		=> all_inputs,
			selected_input		=> selected_input,
			input_index		=> input_index,
			stop			=> stop
		);


end architecture behaviour;
