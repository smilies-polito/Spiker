library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity input_buffer_tb is
end entity input_buffer_tb;

architecture test of input_buffer_tb is

	constant addr_length	: integer := 2;
	constant data_bit_width	: integer := 8;
	constant N_data		: integer := 3;

	-- control input
	signal clk		: std_logic;
	signal load_data	: std_logic;
	signal data_addr	: std_logic_vector(addr_length-1 downto 0);

	-- data input
	signal data_in		: unsigned(data_bit_width-1 downto 0);

	-- data output
	signal data_out		: unsigned(N_data*data_bit_width-1 downto 0);

	component input_buffer is

		generic(
			addr_length	: integer := 10;
			data_bit_width	: integer := 8;
			N_data		: integer := 784
		);

		port(
			-- control input
			clk		: in std_logic;
			load_data	: in std_logic;
			data_addr	: in std_logic_vector(addr_length-1 downto 0);

			-- data input
			data_in		: in unsigned(data_bit_width-1 downto 0);

			-- data output
			data_out	: out unsigned(N_data*data_bit_width-1 downto 0)
		);

	end component input_buffer;


begin

	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;


	-- load data
	load_data_gen	: process
	begin
		load_data <= '0';
		wait for 42 ns;
		load_data <= '0';
		wait;
	end process load_data_gen;


	-- data_addr
	addr_and_data_gen	: process
	begin

		wait for 62 ns;

		for i in 0 to N_data
		loop

			data_addr <= std_logic_vector(to_unsigned(i,
				     data_addr'length));
			data_in <= to_unsigned(i, data_in'length);

			wait for 20 ns;

		end loop;

		wait;

	end process addr_and_data_gen;


	dut	: input_buffer 

		generic map(
			addr_length	=> addr_length,
			data_bit_width	=> data_bit_width,
			N_data		=> N_data
		)

		port map(
			-- control input
			clk		=> clk,
			load_data	=> load_data,
			data_addr	=> data_addr,

			-- data input
			data_in		=> data_in,

			-- data output
			data_out	=> data_out
		);

end architecture test;
