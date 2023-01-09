library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_mux_signed_tb is
end entity generic_mux_signed_tb;


architecture test of generic_mux_signed_tb is

	constant N_sel	: integer := 2;
	constant N	: integer := 8;

	-- input
	signal input_data	: signed(N*(2**N_sel)-1 downto 0);
	signal sel		: std_logic_vector(N_sel-1 downto 0);

	-- output
	signal output_data	: signed(N-1 downto 0);

	component generic_mux_signed is

		generic(

			-- parallelism of the inputs selector
			N_sel	: integer := 4;

			-- parallelism
			N		: integer := 16
				
		);

		port(
			-- input
			input_data	: in signed(N*(2**N_sel)-1 downto 0);
			sel		: in std_logic_vector(N_sel-1 downto 0);

			-- output
			output_data	: out signed(N-1 downto 0)
		);

	end component generic_mux_signed;


begin


	sel_gen		: process
	begin

		for i in 0 to 2**N_sel-1
		loop
			sel	<= std_logic_vector(to_unsigned(i, N_sel));
			wait for 10 ns;

		end loop;

	end process sel_gen;

	
	input_data_gen		: process
	begin

		for i in 0 to 2**N-1
		loop

			for j in 0 to 2**N_sel-1
			loop
				input_data((j+1)*N-1 downto j*N)	<=
				to_signed(i+j, N);
			end loop;

			wait for 40 ns;

		end loop;

	end process input_data_gen;



	dut	: generic_mux_signed 

		generic map(

			-- parallelism of the inputs selector
			N_sel		=> N_sel,

			-- parallelism
			N		=> N
				
		)

		port map(
			-- input
			input_data	=> input_data,
			sel		=> sel,

			-- output
			output_data	=> output_data
		);


end architecture test;
