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


entity rom_128x128_inh is
    port (
        clka : in std_logic;
        addra : in std_logic_vector(6 downto 0);
        dout_0 : out std_logic_vector(0 downto 0);
        dout_1 : out std_logic_vector(0 downto 0);
        dout_2 : out std_logic_vector(0 downto 0);
        dout_3 : out std_logic_vector(0 downto 0);
        dout_4 : out std_logic_vector(0 downto 0);
        dout_5 : out std_logic_vector(0 downto 0);
        dout_6 : out std_logic_vector(0 downto 0);
        dout_7 : out std_logic_vector(0 downto 0);
        dout_8 : out std_logic_vector(0 downto 0);
        dout_9 : out std_logic_vector(0 downto 0);
        dout_a : out std_logic_vector(0 downto 0);
        dout_b : out std_logic_vector(0 downto 0);
        dout_c : out std_logic_vector(0 downto 0);
        dout_d : out std_logic_vector(0 downto 0);
        dout_e : out std_logic_vector(0 downto 0);
        dout_f : out std_logic_vector(0 downto 0);
        dout_10 : out std_logic_vector(0 downto 0);
        dout_11 : out std_logic_vector(0 downto 0);
        dout_12 : out std_logic_vector(0 downto 0);
        dout_13 : out std_logic_vector(0 downto 0);
        dout_14 : out std_logic_vector(0 downto 0);
        dout_15 : out std_logic_vector(0 downto 0);
        dout_16 : out std_logic_vector(0 downto 0);
        dout_17 : out std_logic_vector(0 downto 0);
        dout_18 : out std_logic_vector(0 downto 0);
        dout_19 : out std_logic_vector(0 downto 0);
        dout_1a : out std_logic_vector(0 downto 0);
        dout_1b : out std_logic_vector(0 downto 0);
        dout_1c : out std_logic_vector(0 downto 0);
        dout_1d : out std_logic_vector(0 downto 0);
        dout_1e : out std_logic_vector(0 downto 0);
        dout_1f : out std_logic_vector(0 downto 0);
        dout_20 : out std_logic_vector(0 downto 0);
        dout_21 : out std_logic_vector(0 downto 0);
        dout_22 : out std_logic_vector(0 downto 0);
        dout_23 : out std_logic_vector(0 downto 0);
        dout_24 : out std_logic_vector(0 downto 0);
        dout_25 : out std_logic_vector(0 downto 0);
        dout_26 : out std_logic_vector(0 downto 0);
        dout_27 : out std_logic_vector(0 downto 0);
        dout_28 : out std_logic_vector(0 downto 0);
        dout_29 : out std_logic_vector(0 downto 0);
        dout_2a : out std_logic_vector(0 downto 0);
        dout_2b : out std_logic_vector(0 downto 0);
        dout_2c : out std_logic_vector(0 downto 0);
        dout_2d : out std_logic_vector(0 downto 0);
        dout_2e : out std_logic_vector(0 downto 0);
        dout_2f : out std_logic_vector(0 downto 0);
        dout_30 : out std_logic_vector(0 downto 0);
        dout_31 : out std_logic_vector(0 downto 0);
        dout_32 : out std_logic_vector(0 downto 0);
        dout_33 : out std_logic_vector(0 downto 0);
        dout_34 : out std_logic_vector(0 downto 0);
        dout_35 : out std_logic_vector(0 downto 0);
        dout_36 : out std_logic_vector(0 downto 0);
        dout_37 : out std_logic_vector(0 downto 0);
        dout_38 : out std_logic_vector(0 downto 0);
        dout_39 : out std_logic_vector(0 downto 0);
        dout_3a : out std_logic_vector(0 downto 0);
        dout_3b : out std_logic_vector(0 downto 0);
        dout_3c : out std_logic_vector(0 downto 0);
        dout_3d : out std_logic_vector(0 downto 0);
        dout_3e : out std_logic_vector(0 downto 0);
        dout_3f : out std_logic_vector(0 downto 0);
        dout_40 : out std_logic_vector(0 downto 0);
        dout_41 : out std_logic_vector(0 downto 0);
        dout_42 : out std_logic_vector(0 downto 0);
        dout_43 : out std_logic_vector(0 downto 0);
        dout_44 : out std_logic_vector(0 downto 0);
        dout_45 : out std_logic_vector(0 downto 0);
        dout_46 : out std_logic_vector(0 downto 0);
        dout_47 : out std_logic_vector(0 downto 0);
        dout_48 : out std_logic_vector(0 downto 0);
        dout_49 : out std_logic_vector(0 downto 0);
        dout_4a : out std_logic_vector(0 downto 0);
        dout_4b : out std_logic_vector(0 downto 0);
        dout_4c : out std_logic_vector(0 downto 0);
        dout_4d : out std_logic_vector(0 downto 0);
        dout_4e : out std_logic_vector(0 downto 0);
        dout_4f : out std_logic_vector(0 downto 0);
        dout_50 : out std_logic_vector(0 downto 0);
        dout_51 : out std_logic_vector(0 downto 0);
        dout_52 : out std_logic_vector(0 downto 0);
        dout_53 : out std_logic_vector(0 downto 0);
        dout_54 : out std_logic_vector(0 downto 0);
        dout_55 : out std_logic_vector(0 downto 0);
        dout_56 : out std_logic_vector(0 downto 0);
        dout_57 : out std_logic_vector(0 downto 0);
        dout_58 : out std_logic_vector(0 downto 0);
        dout_59 : out std_logic_vector(0 downto 0);
        dout_5a : out std_logic_vector(0 downto 0);
        dout_5b : out std_logic_vector(0 downto 0);
        dout_5c : out std_logic_vector(0 downto 0);
        dout_5d : out std_logic_vector(0 downto 0);
        dout_5e : out std_logic_vector(0 downto 0);
        dout_5f : out std_logic_vector(0 downto 0);
        dout_60 : out std_logic_vector(0 downto 0);
        dout_61 : out std_logic_vector(0 downto 0);
        dout_62 : out std_logic_vector(0 downto 0);
        dout_63 : out std_logic_vector(0 downto 0);
        dout_64 : out std_logic_vector(0 downto 0);
        dout_65 : out std_logic_vector(0 downto 0);
        dout_66 : out std_logic_vector(0 downto 0);
        dout_67 : out std_logic_vector(0 downto 0);
        dout_68 : out std_logic_vector(0 downto 0);
        dout_69 : out std_logic_vector(0 downto 0);
        dout_6a : out std_logic_vector(0 downto 0);
        dout_6b : out std_logic_vector(0 downto 0);
        dout_6c : out std_logic_vector(0 downto 0);
        dout_6d : out std_logic_vector(0 downto 0);
        dout_6e : out std_logic_vector(0 downto 0);
        dout_6f : out std_logic_vector(0 downto 0);
        dout_70 : out std_logic_vector(0 downto 0);
        dout_71 : out std_logic_vector(0 downto 0);
        dout_72 : out std_logic_vector(0 downto 0);
        dout_73 : out std_logic_vector(0 downto 0);
        dout_74 : out std_logic_vector(0 downto 0);
        dout_75 : out std_logic_vector(0 downto 0);
        dout_76 : out std_logic_vector(0 downto 0);
        dout_77 : out std_logic_vector(0 downto 0);
        dout_78 : out std_logic_vector(0 downto 0);
        dout_79 : out std_logic_vector(0 downto 0);
        dout_7a : out std_logic_vector(0 downto 0);
        dout_7b : out std_logic_vector(0 downto 0);
        dout_7c : out std_logic_vector(0 downto 0);
        dout_7d : out std_logic_vector(0 downto 0);
        dout_7e : out std_logic_vector(0 downto 0);
        dout_7f : out std_logic_vector(0 downto 0)
    );
end entity rom_128x128_inh;

architecture behavior of rom_128x128_inh is


    component rom_128x128_inh_ip is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(6 downto 0);
            douta : out std_logic_vector(127 downto 0)
        );
    end component;


    signal douta : std_logic_vector(127 downto 0);

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
    dout_a <= douta(10 downto 10);
    dout_b <= douta(11 downto 11);
    dout_c <= douta(12 downto 12);
    dout_d <= douta(13 downto 13);
    dout_e <= douta(14 downto 14);
    dout_f <= douta(15 downto 15);
    dout_10 <= douta(16 downto 16);
    dout_11 <= douta(17 downto 17);
    dout_12 <= douta(18 downto 18);
    dout_13 <= douta(19 downto 19);
    dout_14 <= douta(20 downto 20);
    dout_15 <= douta(21 downto 21);
    dout_16 <= douta(22 downto 22);
    dout_17 <= douta(23 downto 23);
    dout_18 <= douta(24 downto 24);
    dout_19 <= douta(25 downto 25);
    dout_1a <= douta(26 downto 26);
    dout_1b <= douta(27 downto 27);
    dout_1c <= douta(28 downto 28);
    dout_1d <= douta(29 downto 29);
    dout_1e <= douta(30 downto 30);
    dout_1f <= douta(31 downto 31);
    dout_20 <= douta(32 downto 32);
    dout_21 <= douta(33 downto 33);
    dout_22 <= douta(34 downto 34);
    dout_23 <= douta(35 downto 35);
    dout_24 <= douta(36 downto 36);
    dout_25 <= douta(37 downto 37);
    dout_26 <= douta(38 downto 38);
    dout_27 <= douta(39 downto 39);
    dout_28 <= douta(40 downto 40);
    dout_29 <= douta(41 downto 41);
    dout_2a <= douta(42 downto 42);
    dout_2b <= douta(43 downto 43);
    dout_2c <= douta(44 downto 44);
    dout_2d <= douta(45 downto 45);
    dout_2e <= douta(46 downto 46);
    dout_2f <= douta(47 downto 47);
    dout_30 <= douta(48 downto 48);
    dout_31 <= douta(49 downto 49);
    dout_32 <= douta(50 downto 50);
    dout_33 <= douta(51 downto 51);
    dout_34 <= douta(52 downto 52);
    dout_35 <= douta(53 downto 53);
    dout_36 <= douta(54 downto 54);
    dout_37 <= douta(55 downto 55);
    dout_38 <= douta(56 downto 56);
    dout_39 <= douta(57 downto 57);
    dout_3a <= douta(58 downto 58);
    dout_3b <= douta(59 downto 59);
    dout_3c <= douta(60 downto 60);
    dout_3d <= douta(61 downto 61);
    dout_3e <= douta(62 downto 62);
    dout_3f <= douta(63 downto 63);
    dout_40 <= douta(64 downto 64);
    dout_41 <= douta(65 downto 65);
    dout_42 <= douta(66 downto 66);
    dout_43 <= douta(67 downto 67);
    dout_44 <= douta(68 downto 68);
    dout_45 <= douta(69 downto 69);
    dout_46 <= douta(70 downto 70);
    dout_47 <= douta(71 downto 71);
    dout_48 <= douta(72 downto 72);
    dout_49 <= douta(73 downto 73);
    dout_4a <= douta(74 downto 74);
    dout_4b <= douta(75 downto 75);
    dout_4c <= douta(76 downto 76);
    dout_4d <= douta(77 downto 77);
    dout_4e <= douta(78 downto 78);
    dout_4f <= douta(79 downto 79);
    dout_50 <= douta(80 downto 80);
    dout_51 <= douta(81 downto 81);
    dout_52 <= douta(82 downto 82);
    dout_53 <= douta(83 downto 83);
    dout_54 <= douta(84 downto 84);
    dout_55 <= douta(85 downto 85);
    dout_56 <= douta(86 downto 86);
    dout_57 <= douta(87 downto 87);
    dout_58 <= douta(88 downto 88);
    dout_59 <= douta(89 downto 89);
    dout_5a <= douta(90 downto 90);
    dout_5b <= douta(91 downto 91);
    dout_5c <= douta(92 downto 92);
    dout_5d <= douta(93 downto 93);
    dout_5e <= douta(94 downto 94);
    dout_5f <= douta(95 downto 95);
    dout_60 <= douta(96 downto 96);
    dout_61 <= douta(97 downto 97);
    dout_62 <= douta(98 downto 98);
    dout_63 <= douta(99 downto 99);
    dout_64 <= douta(100 downto 100);
    dout_65 <= douta(101 downto 101);
    dout_66 <= douta(102 downto 102);
    dout_67 <= douta(103 downto 103);
    dout_68 <= douta(104 downto 104);
    dout_69 <= douta(105 downto 105);
    dout_6a <= douta(106 downto 106);
    dout_6b <= douta(107 downto 107);
    dout_6c <= douta(108 downto 108);
    dout_6d <= douta(109 downto 109);
    dout_6e <= douta(110 downto 110);
    dout_6f <= douta(111 downto 111);
    dout_70 <= douta(112 downto 112);
    dout_71 <= douta(113 downto 113);
    dout_72 <= douta(114 downto 114);
    dout_73 <= douta(115 downto 115);
    dout_74 <= douta(116 downto 116);
    dout_75 <= douta(117 downto 117);
    dout_76 <= douta(118 downto 118);
    dout_77 <= douta(119 downto 119);
    dout_78 <= douta(120 downto 120);
    dout_79 <= douta(121 downto 121);
    dout_7a <= douta(122 downto 122);
    dout_7b <= douta(123 downto 123);
    dout_7c <= douta(124 downto 124);
    dout_7d <= douta(125 downto 125);
    dout_7e <= douta(126 downto 126);
    dout_7f <= douta(127 downto 127);


    rom_128x128_inh_ip_instance : rom_128x128_inh_ip
        port map(
            clka => clka,
            addra => addra,
            douta => douta
        );


end architecture behavior;

