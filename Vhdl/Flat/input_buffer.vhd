library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity input_buffer is

	generic(
		addr_length	: integer := 10;
		data_bit_width	: integer := 8;
		N_data		: integer := 784
	);

	port(
		-- control input
		clk		: in std_logic;
		load_data	: in std_logic;
		data_addr	: in std_logic_vector(addr_length-1 downto 0);

		-- data input
		data_in		: in unsigned(data_bit_width-1 downto 0);

		-- data output
		data_out	: out unsigned(N_data*data_bit_width-1 downto 0)
	);

end entity input_buffer;


architecture behaviour of input_buffer is

	signal data_en		: std_logic_vector(2**addr_length-1 downto 0);
	signal decoded_addr	: std_logic_vector(2**addr_length-1 downto 0);


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


	component reg_unsigned is

		generic(
			-- parallelism
			N	: integer	:= 16		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			reg_in	: in unsigned(N-1 downto 0);

			-- outputs
			reg_out	: out unsigned(N-1 downto 0)
		);

	end component reg_unsigned;

begin

	data_en_gen	: process(load_data, decoded_addr)
	begin

		en_gen	: for i in 0 to 2**addr_length-1
		loop

			data_en(i) <= decoded_addr(i) and load_data;

		end loop en_gen;

	end process data_en_gen;


	data_decoder	: decoder 

		generic map(
			N		=> addr_length
		)

		port map(
			-- input
			encoded_in	=> data_addr,

			-- output
			decoded_out	=> decoded_addr
		);


	registers	: for i in 0 to N_data-1
	generate

		data_reg	: reg_unsigned 

			generic map(
				-- parallelism
				N	=> data_bit_width
			)

			port map(	
				-- inputs	
				clk	=> clk,
				en	=> data_en(i),
				reg_in	=> data_in,

				-- outputs
				reg_out	=> data_out((i+1)*data_bit_width-1 
						downto i*data_bit_width)
			);

	end generate registers;


end architecture behaviour;
