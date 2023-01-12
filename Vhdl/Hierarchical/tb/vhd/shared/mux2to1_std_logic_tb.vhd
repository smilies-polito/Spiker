library ieee;
use ieee.std_logic_1164.all;


entity mux2to1_tb is
end entity mux2to1_tb;

architecture test of mux2to1_tb is

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



	signal sel	: std_logic;
	signal in0	: std_logic;
	signal in1	: std_logic;
	signal mux_out	: std_logic;

begin


	simulate : process
	begin
				   
		sel <= '0';
		in0 <= '1';
		in1 <= '0';
				   
		wait for 10 ns;
		
		sel <= '1';
				   
		wait for 10 ns;
		in0 <= '1';
		in1 <= '0';
				   
		wait for 10 ns;
		
		sel <= '0';
				   
		wait for 10 ns;
		in0 <= '1';
		in1 <= '0';
				   
		wait;
				   
	end process simulate;


	mux	: mux2to1_std_logic
		port map(
			-- inputs 	
			sel	=> sel,	 
			in0	=> in0,
			in1	=> in1,
			  
			-- output
			mux_out	=> mux_out
		);	

end architecture test;
