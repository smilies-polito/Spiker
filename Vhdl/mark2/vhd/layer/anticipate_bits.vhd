library ieee;
use ieee.std_logic_1164.all;

entity anticipate_bits is

	generic(
		-- parallelism
		N		: integer := 8		
	);

	port(
		-- input
		clk		: in std_logic;
		bits_en		: in std_logic;
		anticipate	: in std_logic;
		input_bits	: in std_logic_vector(N-1 downto 0);

		-- output
		output_bits	: out std_logic_vector(N-1 downto 0)	
	);

end entity anticipate_bits;


architecture behaviour of anticipate_bits is

	signal delayed_bits	: std_logic_vector(N-1 downto 0);


	component reg is

		generic(
			-- parallelism
			N	: integer	:= 16		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			reg_in	: in std_logic_vector(N-1 downto 0);

			-- outputs
			reg_out	: out std_logic_vector(N-1 downto 0)
		);

	end component reg;



	component mux2to1 is

		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			sel	: in std_logic;
			in0	: in std_logic_vector(N-1 downto 0);
			in1	: in std_logic_vector(N-1 downto 0);

			-- output
			mux_out	: out std_logic_vector(N-1 downto 0)
		);

	end component mux2to1;


begin


	bits_reg	: reg

		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			clk	=> clk,
			en	=> bits_en,
			reg_in	=> input_bits,

			-- outputs
			reg_out	=> delayed_bits
		);


	bits_mux	: mux2to1

		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> anticipate,
			in0	=> delayed_bits,
			in1	=> input_bits,

			-- output
			mux_out	=> output_bits
		);


end architecture behaviour;
