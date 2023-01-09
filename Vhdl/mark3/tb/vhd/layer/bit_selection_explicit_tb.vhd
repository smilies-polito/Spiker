library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bit_selection_explicit_tb is
end entity bit_selection_explicit_tb;



architecture behaviour of bit_selection_explicit_tb is


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

	-- internal signals
	signal cnt_out	: std_logic_vector(N_cnt-1 downto 0);
	signal mux_in	: std_logic_vector(2**(N_cnt-1)-1 downto 0);


	component generic_or is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			or_in	: in std_logic_vector(N-1 downto 0);

			-- output
			or_out	: out std_logic
		);

	end component generic_or;


	component generic_mux_1bit is

		generic(
			N_sel	: integer := 8		
		);

		port(
			-- input
			mux_in	: in std_logic_vector(2**N_sel-1 downto 0);
			mux_sel	: in std_logic_vector(N_sel-1 downto 0);

			-- output
			mux_out	: out std_logic
		);

	end component generic_mux_1bit;


	component cnt is

		generic(
			N		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(N-1 downto 0)		
		);

	end component cnt;


	component cmp_eq is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			in0	: in std_logic_vector(N-1 downto 0);
			in1	: in std_logic_vector(N-1 downto 0);

			-- output
			cmp_out	: out std_logic
		);

	end component cmp_eq;



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



	-- zero padding of the input to fill all the multiplexer's inputs
	mux_in(N_bit-1 downto 0) 		<= input_bits;
	mux_in(2**(N_cnt-1)-1 downto N_bit) 	<= (others => '0');



	input_index	<= cnt_out;

	inputs_or	: generic_or

		generic map(
			N		=> N_bit
		)

		port map (
			-- input
			or_in		=> input_bits,

			-- output
			or_out		=> all_inputs
		);



	inputs_mux	: generic_mux_1bit

		generic map(
			N_sel		=> N_cnt-1	
		)

		port map(
			-- input
			mux_in		=> mux_in,
			mux_sel		=> cnt_out(N_cnt-2 downto 0),

			-- output
			mux_out		=> selected_input
		);


	select_cnt	: cnt

		generic map(
			N		=> N_cnt
		)

		port map(
			-- input
			clk		=> clk,		
			cnt_en		=> select_cnt_en,
			cnt_rst_n	=> select_cnt_rst_n,

			-- output
			cnt_out		=> cnt_out
		);


	end_cmp		: cmp_eq

		generic map(
			N		=> N_cnt
		)

		port map(
			-- input
			in0		=> cnt_out,
			in1		=> N_inputs,

			-- outpu
			cmp_out		=> stop
		);


end architecture behaviour;
