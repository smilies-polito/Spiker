library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity rf_signed is

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
		di		: in std_logic_vector(word_length-1 downto 0);
		rst_n		: in std_logic;
		rdaddr		: in std_logic_vector(rdwr_addr_length-1 
					downto 0);
		rden		: in std_logic;
		wren		: in std_logic;
		wraddr		: in std_logic_vector(rdwr_addr_length-1
					downto 0);

		-- output
		do		: out std_logic_vector(N_weights_per_word*
					weights_bit_width-1 downto 0)
				
	);

end entity rf_signed;



architecture behaviour of rf_signed is

	constant N_words	: integer := 3;

	signal reg_en		: std_logic_vector(N_words-1 downto 0);
	signal reg_sel		: std_logic_vector(N_words-1 downto 0);

	signal reg_out		: std_logic_vector(N_words*word_length-1 
					downto 0);
	signal selected_reg_out	: std_logic_vector(word_length-1 downto 0);
	signal rden_reg_out	: std_logic_vector(word_length-1 downto 0);
	signal rst_n_reg_out	: std_logic_vector(word_length-1 downto 0);
	signal decoded_out	: std_logic_vector(2**rdwr_addr_length-1
					downto 0);

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

	component reg is

		generic(
			-- parallelism
			N	: integer	:= 16		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			reg_in	: in std_logic_vector(N-1 downto 0);

			-- outputs
			reg_out	: out std_logic_vector(N-1 downto 0)
		);

	end component reg;

	component generic_mux is

		generic(
			N_sel		: integer := 8;
			bit_width	: integer := 8
		);

		port(
			-- input
			mux_in	: in std_logic_vector(2**N_sel*bit_width-1
					downto 0);
			mux_sel	: in std_logic_vector(N_sel-1 downto 0);

			-- output
			mux_out	: out std_logic_vector(bit_width-1 downto 0)
		);

	end component generic_mux;


begin

	reg_sel	<= decoded_out(N_words-1 downto 0);

	-- reg_en
	reg_en_gen	: process(reg_sel, wren)
	begin

		masking	: for i in 0 to N_words-1
		loop

			reg_en(i)	<= reg_sel(i) and wren;

		end loop masking;

	end process reg_en_gen;

	-- rden_reg_out
	rden_reg_out_gen	: process(selected_reg_out, wren)
	begin

		masking	: for i in 0 to N_words-1
		loop

			rden_reg_out(i)	<= selected_reg_out(i) and rden;

		end loop masking;

	end process rden_reg_out_gen;

	-- do
	do_gen	: process(rden_reg_out, wren)
	begin

		masking	: for i in 0 to N_words-1
		loop

			do(i)	<= rden_reg_out(i) and not rst_n;

		end loop masking;

	end process do_gen;

	registers	: for i in 0 to N_words-1
	generate

		register_i	: reg
			generic map(
				N	=> word_length	   
			)

			port map(
				-- inputs
				clk	=> clk,
				en	=> reg_en(i),
				reg_in	=> di,

				-- outputs      
				reg_out	=> reg_out((i+1)*word_length-1 downto
						i*word_length)
			);

	end generate registers;


	addr_decoder	: decoder 

		generic map(
			N		=> rdwr_addr_length
		)

		port map(
			-- input
			encoded_in	=> wraddr,

			-- output
			decoded_out	=> decoded_out
					
		);


	output_mux	: generic_mux 

		generic map(
			N_sel				=> rdwr_addr_length,
			bit_width			=> word_length
		)

		port map(
			-- input
			mux_in(2**rdwr_addr_length*
				word_length-1 
				downto 
				N_words*word_length)	=> (others => '0'),
			mux_in(N_words*word_length-1 
				downto 0)		=> reg_out,
			mux_sel				=> rdaddr,

			-- output
			mux_out				=> selected_reg_out
		);



end architecture behaviour;
