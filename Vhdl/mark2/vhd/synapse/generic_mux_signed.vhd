library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_mux_signed is

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

end entity generic_mux_signed;


architecture behaviour of generic_mux_signed is

	type matrix is array(2**N_sel-1 downto 0) of signed(N-1 downto 0);

	signal mux_in	: matrix;

begin

	-- split the input into the proper amount of signals
	split	: process(input_data)
	begin

		for i in 0 to 2**N_sel-1
		loop
			mux_in(i)	<= input_data((i+1)*N-1 downto i*N);
		end loop;
	
		end process split;



	-- multiplexer behaviour
	selection	: process(mux_in, sel)
	begin

		output_data	<= mux_in(to_integer(unsigned(sel)));

	end process selection;

	


end architecture behaviour;
