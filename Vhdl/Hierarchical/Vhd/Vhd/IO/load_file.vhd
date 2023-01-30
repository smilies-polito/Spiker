library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity load_file is

	generic(
		word_length		: integer := 36;
		bram_addr_length	: integer := 6;
		addr_length		: integer := 16;
		N_bram			: integer := 58;
		N_words			: integer := 784;
		weights_filename	: string := "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/spiker/vhdl/mark3/"&
		"hyperparameters/weights.mem"
	);

	port(
		-- input
		clk			: in std_logic;
		rden			: in std_logic;

		-- output
		di			: out std_logic_vector(word_length-1
						downto 0);
		bram_addr		: out std_logic_vector(bram_addr_length
						-1 downto 0);
		wraddr			: out std_logic_vector(addr_length-1
						downto 0);
		wren			: out std_logic
	);

end entity load_file;


architecture behaviour of load_file is
begin

	-- load weights inside the BRAMs
	load_weights	: process(clk, rden)

		file weights_file	: text open read_mode is
						weights_filename;

		variable read_line	: line;
		variable di_var		: std_logic_vector(word_length-1 
						downto 0);
		variable wraddr_var	: integer := 0;
		variable bram_addr_var	: integer := 0;

	begin

		-- Sample on the clock edge
		if clk'event and clk = '1'
		then

			-- Check if it is ok to start to read the file
			if rden = '1'
			then

				-- Check if the file is not ended
				if not endfile(weights_file)
				then

					-- Check if all the bram have been
					-- already filled
					if bram_addr_var < N_bram
					then

						-- Check if the current bram has
						-- been filled
						if wraddr_var < N_words
						then

							-- Read line from file
							readline(weights_file,
								read_line);
							read(read_line, di_var);

							-- Associate line to
							-- data input
							di	<= di_var;

							-- Set bram address
							bram_addr <=
							std_logic_vector(
							to_unsigned(
							bram_addr_var,
							bram_addr'length));
							
							-- Set write address
							wraddr	<= 
							std_logic_vector(
							to_unsigned(wraddr_var,
							wraddr'length));

							-- Enable write
							wren	<= '1';

							-- Update address
							wraddr_var := 
								wraddr_var + 
								1;
							
						else
							-- Reset address
							wraddr_var := 0;

							-- Update bram address
							bram_addr_var:=
								bram_addr_var
								+ 1;

							-- Disable writing
							wren <= '0';

						end if;

					else

						-- Disable writing
						wren <= '0';

					end if;

				else
					-- Disable writing
					wren <= '0';

				end if;

			else

				-- Disable writing
				wren <= '0';

			end if;

		end if;

	end process load_weights;

end architecture behaviour;
