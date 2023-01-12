library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bram_sdp_tb is
end entity bram_sdp_tb;

architecture test of bram_sdp_tb is


	-- read input
	signal rst		: std_logic;
	signal clk		: std_logic;
	signal rden		: std_logic;
	signal regce		: std_logic;
	signal rdaddr		: std_logic_vector(8 downto 0);
	

	-- write input
	signal we		: std_logic_vector(7 downto 0);
	signal wren		: std_logic;
	signal wraddr		: std_logic_vector(8 downto 0);
	signal di		: std_logic_vector(71 downto 0);

	-- output
	signal do		: std_logic_vector(71 downto 0);

	component bram_sdp is

		port(
			-- read input
			rst	: in std_logic;
			rdclk	: in std_logic;
			rden	: in std_logic;
			regce	: in std_logic;
			rdaddr	: in std_logic_vector(8 downto 0);

			-- write input
			we	: in std_logic_vector(7 downto 0);
			wrclk	: in std_logic;
			wren	: in std_logic;
			wraddr	: in std_logic_vector(8 downto 0);
			di	: in std_logic_vector(71 downto 0);

			-- output
			do	: out std_logic_vector(71 downto 0)
		);

	end component bram_sdp;

begin

	init_we	: process(wren)
	begin
		for i in 0 to 7
		loop
			we(i)	<= wren;
		end loop;
	end process init_we;

	rst	<= '0';
	regce	<= '0';

	-- clock
	clock_gen : process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;


	-- wren
	wren_gen	: process
	begin
		wren	<= '0';
		wait for 14 ns;
		wren	<= '1';
		wait;
	end process wren_gen;


	-- wraddr
	wraddr_gen	: process
	begin
		wraddr	<= (others => '0');
		wait for 14 ns;

		for i in 0 to 511
		loop
			wraddr	<= std_logic_vector(to_unsigned(i, 9));
			wait for 12 ns;
		end loop;

		wait;
	end process wraddr_gen;


	-- di
	di_gen	: process
	begin
		di	<= (others => '0');
		wait for 14 ns;

		for i in 0 to 511
		loop
			di	<= std_logic_vector(to_unsigned(i, di'length));
			wait for 12 ns;
		end loop;
		
		wait;

	end process di_gen;


	-- rden
	rden_gen	: process
	begin
		rden	<= '0';
		wait for 14 ns;
		rden	<= '1';
		wait;
	end process rden_gen;


	-- rdaddr
	rdaddr_gen	: process
	begin
		rdaddr	<= (others => '0');
		wait for 26 ns;

		for i in 0 to 511
		loop
			rdaddr	<= std_logic_vector(to_unsigned(i, 9));
			wait for 12 ns;
		end loop;

		wait;

	end process rdaddr_gen;




	dut	: bram_sdp 
		port map(
			-- read input
			rst	=> rst,
			rdclk	=> clk, 
			rden	=> rden,  
			regce	=> regce,
			rdaddr	=> rdaddr,

			-- write input
			we	=> we,
			wrclk	=> clk,
			wren	=> wren,	
			wraddr	=> wraddr,
			di	=> di,

			-- output
			do	=> do
		);

end architecture test;
