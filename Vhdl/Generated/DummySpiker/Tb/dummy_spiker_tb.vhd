---------------------------------------------------------------------------------
-- This is free and unencumbered software released into the public domain.
--
-- Anyone is free to copy, modify, publish, use, compile, sell, or
-- distribute this software, either in source code form or as a compiled
-- binary, for any purpose, commercial or non-commercial, and by any
-- means.
--
-- In jurisdictions that recognize copyright laws, the author or authors
-- of this software dedicate any and all copyright interest in the
-- software to the public domain. We make this dedication for the benefit
-- of the public at large and to the detriment of our heirs and
-- successors. We intend this dedication to be an overt act of
-- relinquishment in perpetuity of all present and future rights to this
-- software under copyright law.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
-- IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
-- OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
-- ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
-- OTHER DEALINGS IN THE SOFTWARE.
--
-- For more information, please refer to <http://unlicense.org/>
---------------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;
library std;
use std.textio.all;
library work;
use work.spiker_pkg.all;


entity dummy_spiker_tb is
end entity dummy_spiker_tb;

architecture behavior of dummy_spiker_tb is


    constant n_cycles : integer := 10;
    constant cycles_cnt_bitwidth : integer := 5;
    constant in_spike_filename : string := "in_spike.txt";

    component dummy_spiker is
        generic (
            n_cycles : integer := 10;
            cycles_cnt_bitwidth : integer := 5
        );
        port (
            clk : in std_logic;
            rst_n : in std_logic;
            start : in std_logic;
            sample_ready : in std_logic;
            ready : out std_logic;
            sample : out std_logic;
            in_spike : in std_logic;
            in_spike_addr : in std_logic_vector(1 downto 0);
            out_spike : out std_logic;
            out_spike_addr : in std_logic_vector(1 downto 0)
        );
    end component;

	component input_interface is
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
	end component input_interface;

	constant input_size	: integer := 100;
	constant output_size	: integer := 10;
	constant in_addr_len	: integer := 7;
	constant out_addr_len	: integer := 4;


    signal clk : std_logic;
    signal rst_n : std_logic;
    signal start : std_logic;
    signal sample_ready : std_logic;
    signal ready : std_logic;
    signal sample : std_logic;
    signal in_spike : std_logic;
    signal in_spike_addr : std_logic_vector(in_addr_len-1 downto 0);
    signal out_spike : std_logic;
    signal out_spike_addr : std_logic_vector(out_addr_len-1 downto 0);
    signal in_spike_rd_en : std_logic;
	
	signal go	: std_logic;
	signal spikes	: std_logic_vector(input_size-1 downto 0);

begin

    sample_ready <= sample;


    clk_gen : process
    begin

        clk <= '0';
        wait for 10 ns;
        clk <= '1';
        wait for 10 ns;

    end process clk_gen;

    rst_n_gen : process
    begin

        rst_n <= '1';
        wait for 15 ns;
        rst_n <= '0';
        wait for 10 ns;
        rst_n <= '1';

        wait;

    end process rst_n_gen;

    go_gen : process
    begin

        go <= '0';
        wait for 100 ns;
        go <= '1';
        wait for 1000 ns;
        go <= '0';

        wait;

    end process go_gen;

    start_gen : process(clk )
    begin

        if clk'event and clk = '1' 
        then

            if ready = '1' and go = '1'
            then

                start <= '1';

            else
                start <= '0';

            end if;

        end if;

    end process start_gen;

    out_spike_addr_gen : process

	    variable addr_var	: integer := 0;

    begin

	if ready = '1' and go = '0'
	then

		addr_var := 0;

		for i in 0 to 10
		loop
			addr_var := addr_var + 1;
			out_spike_addr <= std_logic_vector(to_unsigned(addr_var,
					  out_spike_addr'length));
			wait for 20 ns;
		end loop;

	end if;

    end process out_spike_addr_gen;

    in_spike_rd_en_gen : process
    begin

        in_spike_rd_en<= '1';

        wait;

    end process in_spike_rd_en_gen;

    in_spike_load : process(clk, in_spike_rd_en )
        variable row : line;
        variable read_var : std_logic;
        file in_spike_file : text open read_mode is in_spike_filename;
    begin

        if clk'event and clk = '1' 
        then

            if not endfile(in_spike_file) 
            then

                if in_spike_rd_en = '1' 
                then

                    readline(in_spike_file, row);
                    read(row, read_var);
                    in_spike <= read_var;

                end if;

            end if;

        end if;

    end process in_spike_load;


    dut : dummy_spiker
        generic map(
            n_cycles => n_cycles,
            cycles_cnt_bitwidth => cycles_cnt_bitwidth
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            start => start,
            sample_ready => sample_ready,
            ready => ready,
            sample => sample,
            in_spike => in_spike,
            in_spike_addr => in_spike_addr,
            out_spike => out_spike,
            out_spike_addr => out_spike_addr
        );


	interface	: input_interface
		generic map(
			N		=> input_size,
			addr_len	=> in_addr_len
		)
		port map(
			clk		=> clk,
			rst_n		=> rst_n,
			spikes		=> spikes,
			sample		=> sample,

			read_en		=> in_spike_rd_en,
			sample_ready	=> sample_ready,
			spike		=> in_spike,
			addr		=> in_spike_addr
		);



end architecture behavior;

