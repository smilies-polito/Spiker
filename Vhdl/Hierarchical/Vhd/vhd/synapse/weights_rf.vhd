library ieee;
use ieee.std_logic_1164.all;

entity weights_bram is

	generic(
		word_length		: integer := 36;
		N_weights_per_word	: integer := 2;
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


	component rf is

		generic(
			word_length		: integer := 36;
			N_weights_per_word	: integer := 2;
			rdwr_addr_length	: integer := 10;
			N_neurons		: integer := 400;
			weights_bit_width	: integer := 5;
			N_bram			: integer := 58
		);

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(word_length-1
						downto 0);
			rst_n		: in std_logic;
			rdaddr		: in std_logic_vector(rdwr_addr_length-1 
						downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(rdwr_addr_length-1
						downto 0);

			-- output
			do		: out std_logic_vector(
						word_length-1 downto 0)
					
		);

	end component rf;

	component decoder is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			encoded_in	: in std_logic_vector(N-1 downto 0);

			-- output
			decoded_out	: out  std_logic_vector(2**N -1 
						downto 0)
		);

	end component decoder;


begin


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

		memory_element	: rf

			generic map(
				word_length		=> word_length,
				N_weights_per_word	=> N_weights_per_word,
				rdwr_addr_length	=> rdwr_addr_length,
				N_neurons		=> N_neurons,
				weights_bit_width	=> weights_bit_width,
				N_bram			=> N_bram
			)

			port map(
				-- input
				clk			=> clk,
				di			=> di,
				rst_n			=> rst_n,
				rdaddr			=> rdaddr,
				rden			=> rden,
				wren			=> wren_int(i),
				wraddr			=> wraddr,

				-- output
				do			=> data_out(i)
						
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
