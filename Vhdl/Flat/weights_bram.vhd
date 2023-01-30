library ieee;
use ieee.std_logic_1164.all;

Library UNISIM;
use UNISIM.vcomponents.all;

Library UNIMACRO;
use UNIMACRO.vcomponents.all;


entity weights_bram is

	generic(
		word_length		: integer := 36;
		N_weights_per_word	: integer := 7;
		rdwr_addr_length	: integer := 10;
		we_length		: integer := 4;
 		N_neurons		: integer := 400;
		weights_bit_width	: integer := 5;
		N_bram			: integer := 58;
		bram_sel_length		: integer := 6
	);

	port(
		-- input
		clk		: in std_logic;
		di		: in std_logic_vector(word_length-1 downto 0);
		rst_n		: in std_logic;
		rdaddr		: in std_logic_vector(rdwr_addr_length-1 
					downto 0);
		rden		: in std_logic;
		wren		: in std_logic;
		wraddr		: in std_logic_vector(rdwr_addr_length-1
					downto 0);
		bram_sel	: in std_logic_vector(bram_sel_length-1 
					downto 0);

		-- output
		do		: out std_logic_vector(N_bram*
					N_weights_per_word*
					weights_bit_width-1 
					downto 0)
				
	);

end entity weights_bram;



architecture behaviour of weights_bram is


	type data_matrix is array(N_bram-1 downto 0) of
		std_logic_vector(word_length-1
		downto 0);

	signal data_out	: data_matrix;

	signal wren_int	: std_logic_vector(2**bram_sel_length-1 downto 0);
	signal rst	: std_logic;
	signal we	: std_logic_vector(we_length-1 downto 0);

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
		for i in 0 to we_length-1
		loop
			we(i)	<= wren;
		end loop;
	end process init_we;

	bram_decoder	: decoder
		generic map(
			N		=> bram_sel_length
		)

		port map(
			-- input
			encoded_in	=> bram_sel,

			-- output
			decoded_out	=> wren_int
		);

	complete_memory	: for i in 0 to N_bram-1
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
				WRITE_WIDTH 		=> word_length,

				-- Valid values are 1-72 (37-72 only valid when
				-- BRAM_SIZE="36Kb")
				READ_WIDTH 		=> word_length,     

				-- Optional output register (0 or 1)
				DO_REG 			=> 0, 
				INIT_FILE 		=> "NONE",

				-- Collision check enable "ALL", "WARNING_ONLY",
				-- "GENERATE_X_ONLY" or "NONE" 
				SIM_COLLISION_CHECK 	=> "ALL", 
				
				--  Set/Reset value for port output
				SRVAL 			=> X"000000000", 

				-- Specify "READ_FIRST" for same clock or
				-- synchronous
				-- clocks. Specify "WRITE_FIRST for asynchrononous
				-- clocks on ports
				WRITE_MODE 		=> "READ_FIRST", 
							   
				--  Initial values on output port
				INIT 			=> X"000000000" 
			)


			port map (
				
				-- Output read data port, width defined by
				-- READ_WIDTH parameter
				do 	=> data_out(i),         

				-- Input write data port, width defined by
				-- WRITE_WIDTH parameter
				di 	=> di,         

				-- Input read address, width defined by read
				-- port depth
				rdaddr 	=> rdaddr, 

				-- 1-bit input read clock
				rdclk 	=> clk,   

				-- 1-bit input read port enable
				rden 	=> rden,     

				-- 1-bit input read output register enable
				regce 	=> '0',   

				-- 1-bit input reset 
				rst 	=> rst, 

				-- Input write enable, width defined by write
				-- port depth
				we 	=> we,         

				-- Input write address, width defined by write
				-- port depth
				wraddr 	=> wraddr, 

				-- 1-bit input write clock
				wrclk 	=> clk,   

				-- 1-bit input write port enable
				wren 	=> wren_int(i)
			);

	end generate complete_memory;


	connect_output	: process(data_out)
	begin

		for i in 0 to N_bram-1
		loop

			do(
				(i+1)*N_weights_per_word*weights_bit_width-1 
				downto
				i*N_weights_per_word*weights_bit_width
			) <=
			data_out(i)(
				N_weights_per_word*weights_bit_width-1
				downto 
				0
			);

		end loop;

	end process connect_output;


end architecture behaviour;
