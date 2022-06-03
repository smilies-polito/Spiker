library ieee;
use ieee.std_logic_1164.all;

Library UNISIM;
use UNISIM.vcomponents.all;

Library UNIMACRO;
use UNIMACRO.vcomponents.all;


entity weights_bram is

	port(
		-- input
		clk		: in std_logic;
		di		: in std_logic_vector(35 downto 0);
		rst_n		: in std_logic;
		rdaddr		: in std_logic_vector(9 downto 0);
		rden		: in std_logic;
		wren		: in std_logic;
		wraddr		: in std_logic_vector(9 downto 0);
		bram_sel	: in std_logic_vector(5 downto 0);

		-- output
		do		: out std_logic_vector(400*5-1 downto 0)
				
	);

end entity weights_bram;



architecture behaviour of weights_bram is

	type data_matrix is array(57 downto 0) of std_logic_vector(35 downto 0);

	signal wren_int	: std_logic_vector(63 downto 0);
	signal data_out	: data_matrix;
	signal rst	: std_logic;
	signal we	: std_logic_vector(3 downto 0);

	component decoder is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			encoded_in	: in std_logic_vector(N-1 downto 0);

			-- output
			decoded_out	: out  std_logic_vector(2**N -1 downto 0)
		);

	end component decoder;


begin

	rst <= not rst_n;


	init_we	: process(wren)
	begin
		for i in 0 to 3
		loop
			we(i)	<= wren;
		end loop;
	end process init_we;

	bram_decoder	: decoder
		generic map(
			N		=> 6
		)

		port map(
			-- input
			encoded_in	=> bram_sel,

			-- output
			decoded_out	=> wren_int
		);

	complete_memory	: for i in 0 to 57
	generate


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
				SRVAL 			=> X"000000000", 

				-- Specify "READ_FIRST" for same clock or synchronous
				-- clocks. Specify "WRITE_FIRST for asynchrononous
				-- clocks on ports
				WRITE_MODE 		=> "READ_FIRST", 
							   
				--  Initial values on output port
				INIT 			=> X"000000000" 
			)


			port map (
				
				-- Output read data port, width defined by READ_WIDTH
				-- parameter
				do 	=> data_out(i),         

				-- Input write data port, width defined by WRITE_WIDTH
				-- parameter
				di 	=> di,         

				-- Input read address, width defined by read port depth
				rdaddr 	=> rdaddr, 

				-- 1-bit input read clock
				rdclk 	=> clk,   

				-- 1-bit input read port enable
				rden 	=> rden,     

				-- 1-bit input read output register enable
				regce 	=> '0',   

				-- 1-bit input reset 
				rst 	=> rst, 

				-- Input write enable, width defined by write port depth
				we 	=> we,         

				-- Input write address, width defined by write port
				-- depth
				wraddr 	=> wraddr, 

				-- 1-bit input write clock
				wrclk 	=> clk,   

				-- 1-bit input write port enable
				wren 	=> wren_int(i)
			);

	end generate complete_memory;


	connect_output	: process(data_out)
	begin

		for i in 0 to 56
		loop

			do((i+1)*7*5-1 downto i*7*5) <= data_out(i)(7*5-1 
				downto 0);

		end loop;

		do(400*5-1 downto 57*7*5) <= data_out(57)(4 downto 0);

	end process connect_output;


end architecture behaviour;
