library ieee;
use ieee.std_logic_1164.all;


entity out_interface_tb is
end entity out_interface_tb;


architecture behaviour of out_interface_tb is

	constant bit_width	: integer := 16;	
	constant N_neurons	: integer := 2;

	type bit_vector_2D is array(N_neurons-1 downto 0) of
		std_logic_vector(bit_width-1 downto 0);

	-- control input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal stop		: std_logic;

	-- data input
	signal spikes		: std_logic_vector(N_neurons-1 downto 0);

	-- output
	signal cnt_out		: std_logic_vector(N_neurons*bit_width-1
				downto 0);

	signal cnt_array	: bit_vector_2D;


	component out_interface is

		generic(
			bit_width	: integer := 16;	
			N_neurons	: integer := 400
		);

		port(
			-- control input
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;

			-- data input
			spikes		: in std_logic_vector(N_neurons-1 downto 0);

			-- output
			cnt_out		: out std_logic_vector(N_neurons*bit_width-1
						downto 0)

		);

	end component out_interface;

begin

	-- split the counters to better visualize them
	split_cnt	: process(cnt_out)
	begin
		for i in 0 to N_neurons-1
		loop
			cnt_array(i) <= cnt_out((i+1)*bit_width-1 downto
					i*bit_width);
		end loop;
			
	end process split_cnt;

	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;

	-- start
	start_gen	: process
	begin
		start <= '0';
		wait for 52 ns;
		start <= '1';
		wait for 20 ns;
		start <= '0';
		wait for 100 ns;
		start <= '1';
		wait for 20 ns;
		start <= '0';
		wait;
	end process start_gen;

	-- stop
	stop_gen	: process
	begin
		stop <= '0';
		wait for 132 ns;
		stop <= '1';
		wait for 20 ns;
		stop <= '0';
		wait;
	end process stop_gen;


	-- reset
	rst_n_gen	: process
	begin
		rst_n <= '1';
		wait for 12 ns;
		rst_n <= '0';
		wait for 5 ns;
		rst_n <= '1';
		wait;
	end process rst_n_gen;

	-- output spikes
	spikes_gen	: process
	begin
		spikes <= "11";
		wait for 92 ns;
		spikes <= "01";
		wait for 20 ns;
		spikes <= "11";
		wait for 20 ns;
		spikes <= "00";
		wait;
	end process spikes_gen;



	dut	: out_interface 

		generic map(
			bit_width	=> bit_width,
			N_neurons	=> N_neurons
		)

		port map(
			-- control input
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,

			-- data input
			spikes		=> spikes,

			-- output
			cnt_out		=> cnt_out
					

		);

	

end architecture behaviour;
