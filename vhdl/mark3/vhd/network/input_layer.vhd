library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity input_layer is

	port(
		-- input
		clk		: in std_logic;
		en		: in std_logic;
		load_n		: in std_logic;
		reg_addr	: in std_logic_vector(9 downto 0);
		pixel_in	: in std_logic_vector(7 downto 0);
		lfsr_in		: in std_logic_vector(12 downto 0);

		-- output
		spikes		: out std_logic_vector(783 downto 0)
	);

end entity input_layer;



architecture behaviour of input_layer is

	type pixel_matrix is array(0 to 783) of std_logic_vector(7 downto 0);
	signal pixels_sig	: pixel_matrix;

	signal reg_en		: std_logic_vector(0 to 1023);
	signal lfsr_out		: std_logic_vector(12 downto 0);


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



	component lfsr_13bit is

		port(
			-- input
			clk		: in std_logic;
			en		: in std_logic;
			load_n		: in std_logic;
			lfsr_in		: in std_logic_vector(12 downto 0);

			-- output
			lfsr_out	: out std_logic_vector(12 downto 0)
		);

	end component lfsr_13bit;


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


	lfsr		: lfsr_13bit 

		port map(
			-- input
			clk		=> clk,
			en		=> en,
			load_n		=> load_n,
			lfsr_in		=> lfsr_in,

			-- output
			lfsr_out	=> lfsr_out
		);


	reg_decoder		: decoder

		generic map(
			N			=> 10	
		)

		port map(
			-- input
			encoded_in		=> reg_addr,

			-- output
			decoded_out		=> reg_en
		);



	registers	: for i in 0 to 783
	generate

		reg_i	: reg
			generic map(
				-- parallelism
				N	=> 8	
			)

			port map (	
				-- inputs	
				clk	=> clk,	
				en	=> reg_en(i),
				reg_in	=> pixel_in,

				-- outputs
				reg_out	=> pixels_sig(i)
			);

	end generate registers;



	cmp_layer	: for i in 0 to 783
	generate
	
		cmp	: unsigned_cmp_gt 
		
			generic map(
				N			=> 13 
			)


			port map(
				-- input
				in0			=> unsigned(lfsr_out),
				in1(12 downto 8)	=> (others => '0'),
				in1(7 downto 0)		=> unsigned(pixels_sig(i)),

				-- output		
				cmp_out			=> spikes(i)
			);

	end generate cmp_layer;



end architecture behaviour;
