library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity unsigned_cmp_gt_tb is
end entity unsigned_cmp_gt_tb;


architecture test of unsigned_cmp_gt_tb is

	constant N	: integer := 8;

	signal in0	: unsigned(N-1 downto 0);
	signal in1	: unsigned(N-1 downto 0);
	signal cmp_out	: std_logic;

	component unsigned_cmp_gt is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			in0	: in unsigned(N-1 downto 0);
			in1	: in unsigned(N-1 downto 0);

			-- output
			cmp_out	: out std_logic
		);

	end component unsigned_cmp_gt;

begin


	simulate	: process
	begin

		in0	<= "00000001";
		in1	<= "00000000";

		wait for 10 ns;

		in0	<= "00000001";
		in1	<= "00001000";

		wait for 10 ns;

		in0	<= "10000000";
		in1	<= "00000000";

		wait for 10 ns;

		in0	<= "10001000";
		in1	<= "10000000";

		wait;


	end process simulate;



	dut	: unsigned_cmp_gt
		generic map(
			N	=> 8		
		)

		port map(
			-- input	
                        in0	=> in0,	
                        in1	=> in1,
                                
                        -- output
			cmp_out	=> cmp_out
		);


end architecture test;
