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


entity rom_10x10_inh is
    port (
        clka : in std_logic;
        addra : in std_logic_vector(3 downto 0);
        dout_0 : out std_logic_vector(0 downto 0);
        dout_1 : out std_logic_vector(0 downto 0);
        dout_2 : out std_logic_vector(0 downto 0);
        dout_3 : out std_logic_vector(0 downto 0);
        dout_4 : out std_logic_vector(0 downto 0);
        dout_5 : out std_logic_vector(0 downto 0);
        dout_6 : out std_logic_vector(0 downto 0);
        dout_7 : out std_logic_vector(0 downto 0);
        dout_8 : out std_logic_vector(0 downto 0);
        dout_9 : out std_logic_vector(0 downto 0)
    );
end entity rom_10x10_inh;

architecture behavior of rom_10x10_inh is


    component rom_10x10_inh_ip is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(3 downto 0);
            douta : out std_logic_vector(9 downto 0)
        );
    end component;


    signal douta : std_logic_vector(9 downto 0);

begin

    dout_0 <= douta(0 downto 0);
    dout_1 <= douta(1 downto 1);
    dout_2 <= douta(2 downto 2);
    dout_3 <= douta(3 downto 3);
    dout_4 <= douta(4 downto 4);
    dout_5 <= douta(5 downto 5);
    dout_6 <= douta(6 downto 6);
    dout_7 <= douta(7 downto 7);
    dout_8 <= douta(8 downto 8);
    dout_9 <= douta(9 downto 9);


    rom_10x10_inh_ip_instance : rom_10x10_inh_ip
        port map(
            clka => clka,
            addra => addra,
            douta => douta
        );


end architecture behavior;

