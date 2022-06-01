library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity generic_reg_file_signed_tb is
end entity generic_reg_file_signed_tb; 


architecture behaviour of generic_reg_file_signed_tb is

	
	constant N_registers	: integer := 3;
	constant N_address	: integer := 2;
	constant N		: integer := 8;


	-- input
	signal clk		: std_logic;
	signal address		: std_logic_vector(N_address-1 downto 0);
	signal wr_en		: std_logic;
	signal input_data	: signed(N-1 downto 0); 
	
	-- output
	signal output_data	: signed(N-1 downto 0);



	signal reg_sel	: std_logic_vector(2**N_address-1 downto 0);
	signal reg_en	: std_logic_vector(2**N_address-1 downto 0);
	signal reg_out	: signed(N*(2**N_address)-1 downto 0);

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


	component generic_mux_signed is

		generic(

			-- parallelism of the inputs selector
			N_sel	: integer := 4;

			-- parallelism
			N		: integer := 16
				
		);

		port(
			-- input
			input_data	: in signed(N*(2**N_sel)-1 downto 0);
			sel		: in std_logic_vector(N_sel-1 downto 0);

			-- output
			output_data	: out signed(N-1 downto 0)
		);

	end component generic_mux_signed;


	component reg_signed is

		generic(
			-- parallelism
			N	: integer	:= 16		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			reg_in	: in signed(N-1 downto 0);

			-- outputs
			reg_out	: out signed(N-1 downto 0)
		);

	end component reg_signed;



begin


	-- clock
	clock_gen	: process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- wr_en
	wr_en_gen	: process
	begin
		wr_en	<= '1';		-- 0 ns
		wait for 38 ns;
		wr_en	<= '0';		-- 38 ns
		wait;
	end process wr_en_gen;


	-- address
	address_gen		: process
	begin

		for i in 0 to N_registers-1
		loop
			address	<= std_logic_vector(to_unsigned(i, N_address));
			wait for 12 ns;

		end loop;

	end process address_gen;


	-- input data
	input_data_gen		: process
	begin

		for i in 0 to N_registers-1
		loop
			input_data	<= to_signed(i, N);
			wait for 12 ns;

		end loop;

		wait;

	end process input_data_gen;



	


	reg_out((2**N_address)*N-1 downto N_registers*N)	<= (others => '0');

	enable_signals	: process(reg_sel, wr_en)
	begin

		for i in 0 to 2**N_address-1 
		loop
			reg_en(i) <= wr_en and reg_sel(i);
		end loop;

	end process enable_signals;



	input_decoder	: decoder

		generic map(
			N		=> N_address		
		)

		port map(
			-- input
			encoded_in	=> address,

			-- output
			decoded_out	=> reg_sel
		);



	output_mux	: generic_mux_signed 

		generic map(

			-- parallelism of the inputs selector
			N_sel		=> N_address,

			-- parallelism
			N		=> N
				
		)

		port map(
			-- input
			input_data	=> reg_out,
			sel		=> address,

			-- output
			output_data	=> output_data
		);



	registers	: for i in 0 to N_registers-1
	generate

		reg_i	: reg_signed
			generic map(
				-- parallelism
				N	=> N
			)

			port map(	
				-- inputs
				clk	=> clk,
				en	=> reg_en(i),
				reg_in	=> input_data,

				-- outputs
				reg_out	=> reg_out((i+1)*N-1 downto i*N)
		);


	end generate registers;



end architecture behaviour;
