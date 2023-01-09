library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity mux2to1_tb is
end entity mux2to1_tb;

architecture test of mux2to1_tb is

	component mux2to1 is
		
		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			sel	: in std_logic;
			in0	: in signed(N-1 downto 0);
			in1	: in signed(N-1 downto 0);

			-- output
			mux_out	: out signed(N-1 downto 0)
		);

	end component mux2to1;


	constant N	: integer := 8;

	signal sel	: std_logic;
	signal in0	: signed(N-1 downto 0);
	signal in1	: signed(N-1 downto 0);
	signal mux_out	: signed(N-1 downto 0);

begin


	simulate : process
	begin

		sel <= '0';
		in0 <= "00001111";
		in1 <= "11110000";

		wait for 10 ns;
		
		sel <= '1';

		wait for 10 ns;
		in0 <= "11111111";
		in1 <= "00000000";

		wait for 10 ns;
		
		sel <= '0';

		wait for 10 ns;
		in0 <= "00001111";
		in1 <= "11110000";

		wait;

	end process simulate;


	mux	: mux2to1
		generic map(
			N	=> 8		
		)
		port map(
			-- inputs 	
			sel	=> sel,	 
			in0	=> in0,
			in1	=> in1,
			  
			-- output
			mux_out	=> mux_out
		);	

end architecture test;
