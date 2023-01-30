library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity generic_mux_1bit_tb is
end entity generic_mux_1bit_tb;


architecture test of generic_mux_1bit_tb is

	constant N_sel	: integer := 2;

	signal mux_in	: std_logic_vector(2**N_sel-1 downto 0);
	signal mux_sel	: std_logic_vector(N_sel-1 downto 0);
	signal mux_out	: std_logic;


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



begin



	mux_sel_gen	: process
	begin

		for i in 0 to 2**N_sel-1
		loop

			mux_sel	<= std_logic_vector(to_unsigned(i, N_sel));
			wait for 20 ns;

		end loop;

		wait;

	end process mux_sel_gen;


	mux_in_gen	: process
	begin

		for i in 0 to 2**N_sel-1
		loop
			mux_in		<= (others => '0');
			wait for 10 ns;
			mux_in(i)	<= '1';
			wait for 10 ns;

		end loop;

		mux_in		<= (others => '0');
		wait;

	end process mux_in_gen;




	dut	: generic_mux_1bit 

		generic map(
			N_sel	=> N_sel	
		)

		port map(
			-- input
			mux_in	=> mux_in,
			mux_sel	=> mux_sel,

			-- outpu=>t
			mux_out	=> mux_out	
		);


end architecture test;
