library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity input_interface is
	generic(
		N		: integer := 100;
		addr_len	: integer := 7
	);
	port(
		clk		: in std_logic;
		rst_n		: in std_logic;
		spikes		: in std_logic_vector(N-1 downto 0);
		sample		: in std_logic;

		read_en		: out std_logic;
		sample_ready	: out std_logic;
		spike		: out std_logic;
		addr		: out std_logic_vector(addr_len-1 downto 0)
	);
end entity input_interface;

architecture behavior of input_interface is

	signal spikes_end	: std_logic_vector(addr_len-1 downto 0);
	signal cnt_en		: std_logic;
	signal cnt_rst_n	: std_logic;
	signal cnt		: std_logic_vector(addr_len-1 downto 0);
	signal load_end		: std_logic;

	type states is (
		reset,
		idle,
		read,
		load
	); 

	signal present_state, next_state	: states;

begin
	addr		<= cnt;
	spikes_end 	<= std_logic_vector(to_unsigned(N, addr_len));
	spike		<= spikes(to_integer(unsigned(cnt)));

	terminal_counter : process(spikes_end, cnt)
	begin

		if cnt = spikes_end
		then
			load_end <= '1';
		else
			load_end <= '0';
		end if;

	end process terminal_counter;


	count	: process(clk)

		variable cnt_var	: integer	:= 0;
	begin

		if clk'event and clk = '1'
		then
			if cnt_rst_n = '0'
			then
				cnt_var	:= 0;
			else
				if cnt_en = '1'
				then
					cnt_var	:= cnt_var + 1;
				end if;
			end if;
		end if;

		cnt	<= std_logic_vector(to_unsigned(cnt_var, addr_len));

	end process count;


	state_transition	: process(clk, rst_n)
	begin

		if rst_n = '0'
		then
			present_state 	<= reset;
		elsif clk'event and clk = '1'
		then
			present_state	<= next_state;
		end if;

	end process state_transition;

	state_evaluation	: process(present_state, sample, load_end)
	begin

		case present_state is
			when reset =>
				next_state <= idle;
			when idle =>
				if sample = '0'
				then
					next_state <= idle;
				else
				next_state <= read;
				end if;

			when read =>
				next_state <= load;
			when load =>
				if load_end = '1'
				then
					next_state <= idle;
				else
					next_state <= load;
				end if;
			when others =>
				next_state <= reset;
		end case;

	end process state_evaluation;

	output_evaluation	: process(present_state)
	begin

		case present_state is
			when reset =>
				cnt_rst_n	<= '0';

			when idle =>
				sample_ready 	<= '1';
				cnt_rst_n	<= '0';

			when read =>
				read_en		<= '1';

			when load =>
				cnt_en		<= '1';

			when others =>
				cnt_rst_n	<= '0';

		end case;

	end process output_evaluation;


end architecture behavior;
