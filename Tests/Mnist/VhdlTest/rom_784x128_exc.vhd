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


entity rom_784x128_exc is
    port (
        clka : in std_logic;
        addra : in std_logic_vector(9 downto 0);
        dout_0 : out std_logic_vector(3 downto 0);
        dout_1 : out std_logic_vector(3 downto 0);
        dout_2 : out std_logic_vector(3 downto 0);
        dout_3 : out std_logic_vector(3 downto 0);
        dout_4 : out std_logic_vector(3 downto 0);
        dout_5 : out std_logic_vector(3 downto 0);
        dout_6 : out std_logic_vector(3 downto 0);
        dout_7 : out std_logic_vector(3 downto 0);
        dout_8 : out std_logic_vector(3 downto 0);
        dout_9 : out std_logic_vector(3 downto 0);
        dout_a : out std_logic_vector(3 downto 0);
        dout_b : out std_logic_vector(3 downto 0);
        dout_c : out std_logic_vector(3 downto 0);
        dout_d : out std_logic_vector(3 downto 0);
        dout_e : out std_logic_vector(3 downto 0);
        dout_f : out std_logic_vector(3 downto 0);
        dout_10 : out std_logic_vector(3 downto 0);
        dout_11 : out std_logic_vector(3 downto 0);
        dout_12 : out std_logic_vector(3 downto 0);
        dout_13 : out std_logic_vector(3 downto 0);
        dout_14 : out std_logic_vector(3 downto 0);
        dout_15 : out std_logic_vector(3 downto 0);
        dout_16 : out std_logic_vector(3 downto 0);
        dout_17 : out std_logic_vector(3 downto 0);
        dout_18 : out std_logic_vector(3 downto 0);
        dout_19 : out std_logic_vector(3 downto 0);
        dout_1a : out std_logic_vector(3 downto 0);
        dout_1b : out std_logic_vector(3 downto 0);
        dout_1c : out std_logic_vector(3 downto 0);
        dout_1d : out std_logic_vector(3 downto 0);
        dout_1e : out std_logic_vector(3 downto 0);
        dout_1f : out std_logic_vector(3 downto 0);
        dout_20 : out std_logic_vector(3 downto 0);
        dout_21 : out std_logic_vector(3 downto 0);
        dout_22 : out std_logic_vector(3 downto 0);
        dout_23 : out std_logic_vector(3 downto 0);
        dout_24 : out std_logic_vector(3 downto 0);
        dout_25 : out std_logic_vector(3 downto 0);
        dout_26 : out std_logic_vector(3 downto 0);
        dout_27 : out std_logic_vector(3 downto 0);
        dout_28 : out std_logic_vector(3 downto 0);
        dout_29 : out std_logic_vector(3 downto 0);
        dout_2a : out std_logic_vector(3 downto 0);
        dout_2b : out std_logic_vector(3 downto 0);
        dout_2c : out std_logic_vector(3 downto 0);
        dout_2d : out std_logic_vector(3 downto 0);
        dout_2e : out std_logic_vector(3 downto 0);
        dout_2f : out std_logic_vector(3 downto 0);
        dout_30 : out std_logic_vector(3 downto 0);
        dout_31 : out std_logic_vector(3 downto 0);
        dout_32 : out std_logic_vector(3 downto 0);
        dout_33 : out std_logic_vector(3 downto 0);
        dout_34 : out std_logic_vector(3 downto 0);
        dout_35 : out std_logic_vector(3 downto 0);
        dout_36 : out std_logic_vector(3 downto 0);
        dout_37 : out std_logic_vector(3 downto 0);
        dout_38 : out std_logic_vector(3 downto 0);
        dout_39 : out std_logic_vector(3 downto 0);
        dout_3a : out std_logic_vector(3 downto 0);
        dout_3b : out std_logic_vector(3 downto 0);
        dout_3c : out std_logic_vector(3 downto 0);
        dout_3d : out std_logic_vector(3 downto 0);
        dout_3e : out std_logic_vector(3 downto 0);
        dout_3f : out std_logic_vector(3 downto 0);
        dout_40 : out std_logic_vector(3 downto 0);
        dout_41 : out std_logic_vector(3 downto 0);
        dout_42 : out std_logic_vector(3 downto 0);
        dout_43 : out std_logic_vector(3 downto 0);
        dout_44 : out std_logic_vector(3 downto 0);
        dout_45 : out std_logic_vector(3 downto 0);
        dout_46 : out std_logic_vector(3 downto 0);
        dout_47 : out std_logic_vector(3 downto 0);
        dout_48 : out std_logic_vector(3 downto 0);
        dout_49 : out std_logic_vector(3 downto 0);
        dout_4a : out std_logic_vector(3 downto 0);
        dout_4b : out std_logic_vector(3 downto 0);
        dout_4c : out std_logic_vector(3 downto 0);
        dout_4d : out std_logic_vector(3 downto 0);
        dout_4e : out std_logic_vector(3 downto 0);
        dout_4f : out std_logic_vector(3 downto 0);
        dout_50 : out std_logic_vector(3 downto 0);
        dout_51 : out std_logic_vector(3 downto 0);
        dout_52 : out std_logic_vector(3 downto 0);
        dout_53 : out std_logic_vector(3 downto 0);
        dout_54 : out std_logic_vector(3 downto 0);
        dout_55 : out std_logic_vector(3 downto 0);
        dout_56 : out std_logic_vector(3 downto 0);
        dout_57 : out std_logic_vector(3 downto 0);
        dout_58 : out std_logic_vector(3 downto 0);
        dout_59 : out std_logic_vector(3 downto 0);
        dout_5a : out std_logic_vector(3 downto 0);
        dout_5b : out std_logic_vector(3 downto 0);
        dout_5c : out std_logic_vector(3 downto 0);
        dout_5d : out std_logic_vector(3 downto 0);
        dout_5e : out std_logic_vector(3 downto 0);
        dout_5f : out std_logic_vector(3 downto 0);
        dout_60 : out std_logic_vector(3 downto 0);
        dout_61 : out std_logic_vector(3 downto 0);
        dout_62 : out std_logic_vector(3 downto 0);
        dout_63 : out std_logic_vector(3 downto 0);
        dout_64 : out std_logic_vector(3 downto 0);
        dout_65 : out std_logic_vector(3 downto 0);
        dout_66 : out std_logic_vector(3 downto 0);
        dout_67 : out std_logic_vector(3 downto 0);
        dout_68 : out std_logic_vector(3 downto 0);
        dout_69 : out std_logic_vector(3 downto 0);
        dout_6a : out std_logic_vector(3 downto 0);
        dout_6b : out std_logic_vector(3 downto 0);
        dout_6c : out std_logic_vector(3 downto 0);
        dout_6d : out std_logic_vector(3 downto 0);
        dout_6e : out std_logic_vector(3 downto 0);
        dout_6f : out std_logic_vector(3 downto 0);
        dout_70 : out std_logic_vector(3 downto 0);
        dout_71 : out std_logic_vector(3 downto 0);
        dout_72 : out std_logic_vector(3 downto 0);
        dout_73 : out std_logic_vector(3 downto 0);
        dout_74 : out std_logic_vector(3 downto 0);
        dout_75 : out std_logic_vector(3 downto 0);
        dout_76 : out std_logic_vector(3 downto 0);
        dout_77 : out std_logic_vector(3 downto 0);
        dout_78 : out std_logic_vector(3 downto 0);
        dout_79 : out std_logic_vector(3 downto 0);
        dout_7a : out std_logic_vector(3 downto 0);
        dout_7b : out std_logic_vector(3 downto 0);
        dout_7c : out std_logic_vector(3 downto 0);
        dout_7d : out std_logic_vector(3 downto 0);
        dout_7e : out std_logic_vector(3 downto 0);
        dout_7f : out std_logic_vector(3 downto 0)
    );
end entity rom_784x128_exc;

architecture behavior of rom_784x128_exc is


    component rom_784x128_exc_ip is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(9 downto 0);
            douta : out std_logic_vector(511 downto 0)
        );
    end component;


    signal douta : std_logic_vector(511 downto 0);

begin

    dout_0 <= douta(3 downto 0);
    dout_1 <= douta(7 downto 4);
    dout_2 <= douta(11 downto 8);
    dout_3 <= douta(15 downto 12);
    dout_4 <= douta(19 downto 16);
    dout_5 <= douta(23 downto 20);
    dout_6 <= douta(27 downto 24);
    dout_7 <= douta(31 downto 28);
    dout_8 <= douta(35 downto 32);
    dout_9 <= douta(39 downto 36);
    dout_a <= douta(43 downto 40);
    dout_b <= douta(47 downto 44);
    dout_c <= douta(51 downto 48);
    dout_d <= douta(55 downto 52);
    dout_e <= douta(59 downto 56);
    dout_f <= douta(63 downto 60);
    dout_10 <= douta(67 downto 64);
    dout_11 <= douta(71 downto 68);
    dout_12 <= douta(75 downto 72);
    dout_13 <= douta(79 downto 76);
    dout_14 <= douta(83 downto 80);
    dout_15 <= douta(87 downto 84);
    dout_16 <= douta(91 downto 88);
    dout_17 <= douta(95 downto 92);
    dout_18 <= douta(99 downto 96);
    dout_19 <= douta(103 downto 100);
    dout_1a <= douta(107 downto 104);
    dout_1b <= douta(111 downto 108);
    dout_1c <= douta(115 downto 112);
    dout_1d <= douta(119 downto 116);
    dout_1e <= douta(123 downto 120);
    dout_1f <= douta(127 downto 124);
    dout_20 <= douta(131 downto 128);
    dout_21 <= douta(135 downto 132);
    dout_22 <= douta(139 downto 136);
    dout_23 <= douta(143 downto 140);
    dout_24 <= douta(147 downto 144);
    dout_25 <= douta(151 downto 148);
    dout_26 <= douta(155 downto 152);
    dout_27 <= douta(159 downto 156);
    dout_28 <= douta(163 downto 160);
    dout_29 <= douta(167 downto 164);
    dout_2a <= douta(171 downto 168);
    dout_2b <= douta(175 downto 172);
    dout_2c <= douta(179 downto 176);
    dout_2d <= douta(183 downto 180);
    dout_2e <= douta(187 downto 184);
    dout_2f <= douta(191 downto 188);
    dout_30 <= douta(195 downto 192);
    dout_31 <= douta(199 downto 196);
    dout_32 <= douta(203 downto 200);
    dout_33 <= douta(207 downto 204);
    dout_34 <= douta(211 downto 208);
    dout_35 <= douta(215 downto 212);
    dout_36 <= douta(219 downto 216);
    dout_37 <= douta(223 downto 220);
    dout_38 <= douta(227 downto 224);
    dout_39 <= douta(231 downto 228);
    dout_3a <= douta(235 downto 232);
    dout_3b <= douta(239 downto 236);
    dout_3c <= douta(243 downto 240);
    dout_3d <= douta(247 downto 244);
    dout_3e <= douta(251 downto 248);
    dout_3f <= douta(255 downto 252);
    dout_40 <= douta(259 downto 256);
    dout_41 <= douta(263 downto 260);
    dout_42 <= douta(267 downto 264);
    dout_43 <= douta(271 downto 268);
    dout_44 <= douta(275 downto 272);
    dout_45 <= douta(279 downto 276);
    dout_46 <= douta(283 downto 280);
    dout_47 <= douta(287 downto 284);
    dout_48 <= douta(291 downto 288);
    dout_49 <= douta(295 downto 292);
    dout_4a <= douta(299 downto 296);
    dout_4b <= douta(303 downto 300);
    dout_4c <= douta(307 downto 304);
    dout_4d <= douta(311 downto 308);
    dout_4e <= douta(315 downto 312);
    dout_4f <= douta(319 downto 316);
    dout_50 <= douta(323 downto 320);
    dout_51 <= douta(327 downto 324);
    dout_52 <= douta(331 downto 328);
    dout_53 <= douta(335 downto 332);
    dout_54 <= douta(339 downto 336);
    dout_55 <= douta(343 downto 340);
    dout_56 <= douta(347 downto 344);
    dout_57 <= douta(351 downto 348);
    dout_58 <= douta(355 downto 352);
    dout_59 <= douta(359 downto 356);
    dout_5a <= douta(363 downto 360);
    dout_5b <= douta(367 downto 364);
    dout_5c <= douta(371 downto 368);
    dout_5d <= douta(375 downto 372);
    dout_5e <= douta(379 downto 376);
    dout_5f <= douta(383 downto 380);
    dout_60 <= douta(387 downto 384);
    dout_61 <= douta(391 downto 388);
    dout_62 <= douta(395 downto 392);
    dout_63 <= douta(399 downto 396);
    dout_64 <= douta(403 downto 400);
    dout_65 <= douta(407 downto 404);
    dout_66 <= douta(411 downto 408);
    dout_67 <= douta(415 downto 412);
    dout_68 <= douta(419 downto 416);
    dout_69 <= douta(423 downto 420);
    dout_6a <= douta(427 downto 424);
    dout_6b <= douta(431 downto 428);
    dout_6c <= douta(435 downto 432);
    dout_6d <= douta(439 downto 436);
    dout_6e <= douta(443 downto 440);
    dout_6f <= douta(447 downto 444);
    dout_70 <= douta(451 downto 448);
    dout_71 <= douta(455 downto 452);
    dout_72 <= douta(459 downto 456);
    dout_73 <= douta(463 downto 460);
    dout_74 <= douta(467 downto 464);
    dout_75 <= douta(471 downto 468);
    dout_76 <= douta(475 downto 472);
    dout_77 <= douta(479 downto 476);
    dout_78 <= douta(483 downto 480);
    dout_79 <= douta(487 downto 484);
    dout_7a <= douta(491 downto 488);
    dout_7b <= douta(495 downto 492);
    dout_7c <= douta(499 downto 496);
    dout_7d <= douta(503 downto 500);
    dout_7e <= douta(507 downto 504);
    dout_7f <= douta(511 downto 508);


    rom_784x128_exc_ip_instance : rom_784x128_exc_ip
        port map(
            clka => clka,
            addra => addra,
            douta => douta
        );


end architecture behavior;

