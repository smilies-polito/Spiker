library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity decoder_tb is
end entity decoder_tb;


architecture test of decoder_tb is

	constant N		: integer := 3;

	signal encoded_in	: std_logic_vector(N-1 downto 0);
	signal decoded_out	: std_logic_vector(0 to 2**N-1);

	component decoder is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			encoded_in	: in std_logic_vector(N-1 downto 0);

			-- output
			decoded_out	: out  std_logic_vector(0 to 2**N-1)
		);

	end component decoder;


begin



	input_gen : process
	begin

		for i in 0 to 2**N-1
		loop
			encoded_in	<= std_logic_vector(to_unsigned(i, 3));
			wait for 10 ns;
		end loop;

		wait;
	
	end process input_gen;



	dut: decoder
		generic map(
			N		=> N
		)

		port map(
			-- input
			encoded_in	=> encoded_in,
                                                      
			-- output          
			decoded_out	=> decoded_out
		);


end architecture test;
