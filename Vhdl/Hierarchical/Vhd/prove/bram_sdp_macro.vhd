library ieee;
use ieee.std_logic_1164.all;

Library UNISIM;
use UNISIM.vcomponents.all;

Library UNIMACRO;
use UNIMACRO.vcomponents.all;

entity bram_sdp is

	port(
		-- read input
		rst	: in std_logic;
		rdclk	: in std_logic;
		rden	: in std_logic;
		regce	: in std_logic;
		rdaddr	: in std_logic_vector(9 downto 0);

		-- write input
		we	: in std_logic_vector(3 downto 0);
		wrclk	: in std_logic;
		wren	: in std_logic;
		wraddr	: in std_logic_vector(9 downto 0);
		di	: in std_logic_vector(35 downto 0);

		-- output
		do	: out std_logic_vector(35 downto 0)
	);

end entity bram_sdp;


architecture behaviour of bram_sdp is
begin


   -- BRAM_SDP_MACRO: Simple Dual Port RAM
   --                 Artix-7
   -- Xilinx HDL Language Template, version 2020.2
   
   -- Note: 	This Unimacro model assumes the port directions to be "downto". 
   --         	Simulation of this model with "to" in the port directions could
   --		lead to erroneous results.

   -----------------------------------------------------------------------
   --  READ_WIDTH | BRAM_SIZE | READ Depth  | RDADDR Width |            --
   -- WRITE_WIDTH |           | WRITE Depth | WRADDR Width |  WE Width  --
   -- ============|===========|=============|==============|============--
   --    37-72    |  "36Kb"   |      512    |     9-bit    |    8-bit   --
   --    19-36    |  "36Kb"   |     1024    |    10-bit    |    4-bit   --
   --    19-36    |  "18Kb"   |      512    |     9-bit    |    4-bit   --
   --    10-18    |  "36Kb"   |     2048    |    11-bit    |    2-bit   --
   --    10-18    |  "18Kb"   |     1024    |    10-bit    |    2-bit   --
   --     5-9     |  "36Kb"   |     4096    |    12-bit    |    1-bit   --
   --     5-9     |  "18Kb"   |     2048    |    11-bit    |    1-bit   --
   --     3-4     |  "36Kb"   |     8192    |    13-bit    |    1-bit   --
   --     3-4     |  "18Kb"   |     4096    |    12-bit    |    1-bit   --
   --       2     |  "36Kb"   |    16384    |    14-bit    |    1-bit   --
   --       2     |  "18Kb"   |     8192    |    13-bit    |    1-bit   --
   --       1     |  "36Kb"   |    32768    |    15-bit    |    1-bit   --
   --       1     |  "18Kb"   |    16384    |    14-bit    |    1-bit   --
   -----------------------------------------------------------------------


	BRAM_SDP_MACRO_inst : BRAM_SDP_MACRO
		generic map (

			-- Target BRAM, "18Kb" or "36Kb" 
			BRAM_SIZE 		=> "36Kb", 

			-- Target device: "VIRTEX5", "VIRTEX6", "7SERIES",
			-- "SPARTAN6" 
			DEVICE 			=> "7SERIES", 

			-- Valid values are 1-72 (37-72 only valid when
			-- BRAM_SIZE="36Kb")
			WRITE_WIDTH 		=> 36,

			-- Valid values are 1-72 (37-72 only valid when
			-- BRAM_SIZE="36Kb")
			READ_WIDTH 		=> 36,     

			-- Optional output register (0 or 1)
			DO_REG 			=> 0, 
			INIT_FILE 		=> "NONE",

			-- Collision check enable "ALL", "WARNING_ONLY",
			-- "GENERATE_X_ONLY" or "NONE" 
			SIM_COLLISION_CHECK 	=> "ALL", 
			
			--  Set/Reset value for port output
			SRVAL 			=> X"000000000000000000", 

			-- Specify "READ_FIRST" for same clock or synchronous
			-- clocks. Specify "WRITE_FIRST for asynchrononous
			-- clocks on ports
			WRITE_MODE 		=> "READ_FIRST", 
						   
			--  Initial values on output port
			INIT 			=> X"000000000000000000" 
		)


		port map (
			
			-- Output read data port, width defined by READ_WIDTH
			-- parameter
			DO 	=> DO,         

			-- Input write data port, width defined by WRITE_WIDTH
			-- parameter
			DI 	=> DI,         

			-- Input read address, width defined by read port depth
			RDADDR 	=> RDADDR, 

			-- 1-bit input read clock
			RDCLK 	=> RDCLK,   

			-- 1-bit input read port enable
			RDEN 	=> RDEN,     

			-- 1-bit input read output register enable
			REGCE 	=> REGCE,   

			-- 1-bit input reset 
			RST 	=> RST,       

			-- Input write enable, width defined by write port depth
			WE 	=> WE,         

			-- Input write address, width defined by write port
			-- depth
			WRADDR 	=> WRADDR, 

			-- 1-bit input write clock
			WRCLK 	=> WRCLK,   

			-- 1-bit input write port enable
			WREN 	=> WREN      
		);

end architecture behaviour;
