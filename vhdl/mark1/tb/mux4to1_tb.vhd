library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity mux4to1_tb is
end entity mux4to1_tb;

architecture test of mux4to1_tb is

	component mux4to1 is
		
		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			sel	: in std_logic_vector(1 downto 0);
			in0	: in signed(N-1 downto 0);
			in1	: in signed(N-1 downto 0);
			in2	: in signed(N-1 downto 0);
			in3	: in signed(N-1 downto 0);

			-- output
			mux_out	: out signed(N-1 downto 0)
		);

	end component mux4to1;


	constant N	: integer := 8;

	signal sel	: std_logic_vector(1 downto 0);
	signal in0	: signed(N-1 downto 0);
	signal in1	: signed(N-1 downto 0);
	signal in2	: signed(N-1 downto 0);
	signal in3	: signed(N-1 downto 0);
	signal mux_out	: signed(N-1 downto 0);

begin


	simulate : process
	begin

		sel <= "00";
		in0 <= "00001111";
		in1 <= "11110000";
		in2 <= "11001100";
		in3 <= "00110011";

		wait for 10 ns;
		
		sel <= "01";

		wait for 10 ns;

		sel <= "10";

		wait for 10 ns;

		sel <= "11";

		wait for 10 ns;
	
		sel <= "00";
		
		wait for 10 ns;
		in0 <= "11111111";
		wait for 10 ns;
		in0 <= "10101010";

		wait for 10 ns;
	
		sel <= "01";
		
		wait for 10 ns;
		in1 <= "11111111";
		wait for 10 ns;
		in1 <= "10101010";

		wait for 10 ns;
	
		sel <= "10";
		
		wait for 10 ns;
		in2 <= "11111111";
		wait for 10 ns;
		in2 <= "10101010";

		wait for 10 ns;
	
		sel <= "11";
		
		wait for 10 ns;
		in3 <= "11111111";
		wait for 10 ns;
		in3 <= "10101010";

		wait;

	end process simulate;


	mux	: mux4to1
		generic map(
			N	=> 8		
		)
		port map(
			-- inputs 	
			sel	=> sel,	 
			in0	=> in0,
			in1	=> in1,
			in2	=> in2,
			in3	=> in3,
			  
			-- output
			mux_out	=> mux_out
		);	

end architecture test;
