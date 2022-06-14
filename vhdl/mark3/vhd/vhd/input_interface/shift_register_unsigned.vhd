library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity shift_register_unsigned is

	generic(
		bit_width	: integer := 8		
	);

	port(
		-- input
		clk		: in std_logic;
		shift_in	: in std_logic;
		reg_en		: in std_logic;
		shift_en	: in std_logic;
		reg_in		: in unsigned(bit_width-1 downto 0);

		-- output
		reg_out		: out unsigned(bit_width-1 downto 0)
	);

end entity shift_register_unsigned;



architecture behaviour of shift_register_unsigned is

	signal ff_in	: unsigned(bit_width-1 downto 0);
	signal ff_out	: unsigned(bit_width-1 downto 0);

	component ff is

		port(
			-- input
			clk	: in std_logic;
			ff_en	: in std_logic;		
			ff_in	: in std_logic;

			-- output
			ff_out	: out std_logic
		);

	end component ff;


	component mux2to1_std_logic is

		port(	
			-- inputs	
			sel	: in std_logic;
			in0	: in std_logic;
			in1	: in std_logic;

			-- output
			mux_out	: out std_logic
		);

	end component mux2to1_std_logic;




begin

	reg_out	<= ff_out;

	flip_flops	: for i in 0 to bit_width-1
	generate
	
		flip_flop	: ff
			port map(
				-- input
				clk	=> clk,
				ff_en	=> reg_en,
				ff_in	=> ff_in(i),

				-- output
				ff_out	=> ff_out(i) 
			);

	end generate flip_flops;


	


	multiplexers	: for i in 0 to bit_width-2
	generate

		mux	: mux2to1_std_logic
			port map(
				-- input
				sel	=> shift_en,
                                in0	=> reg_in(i),
                                in1	=> ff_out(i+1),
                                       
                                -- outp
                                mux_out	=> ff_in(i)
			);

	end generate multiplexers;


	mux_n	: mux2to1_std_logic
		port map(
			-- input
			sel	=> shift_en,
			in0	=> reg_in(bit_width-1),
			in1	=> shift_in,
			       
			-- outp
			mux_out	=> ff_in(bit_width-1)
		);

	

end architecture behaviour;
