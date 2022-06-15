library ieee;
use ieee.std_logic_1164.all;

entity bit_selection is

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

end entity bit_selection;



architecture behaviour of bit_selection is

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
			bit_width	: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(bit_width-1 
						downto 0)
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

	-- zero padding of the input to fill all the multiplexer's inputs
	mux_in(N_bit-1 downto 0) 	<= input_bits;
	mux_in(2**(N_cnt-1)-1 downto N_bit) <= (others => '0');



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
			bit_width	=> N_cnt
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
