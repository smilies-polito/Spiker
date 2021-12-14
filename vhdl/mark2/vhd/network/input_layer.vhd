library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity input_layer is

	port(
		-- input
		clk		: in std_logic;
		en		: in std_logic;
		load_n		: in std_logic;
		pixels		: in std_logic_vector(784*8-1 downto 0);

		-- output
		spikes		: out std_logic_vector(783 downto 0)
	);

end entity input_layer;



architecture behaviour of input_layer is

	type matrix is array(783 downto 0) of std_logic_vector(12 downto 0);
	type pixel_matrix is array(783 downto 0) of std_logic_vector(7 downto 0);

	signal lfsr_in		: matrix;
	signal lfsr_out		: matrix;

	signal pixels_sig	: pixel_matrix;


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



	split_pixels	: process(pixels)
	begin
		for i in 0 to 783
		loop
			pixels_sig(i)	<= pixels((i+1)*8-1 downto i*8);
		end loop;
	end process split_pixels;


	lfsr_layer	: for i in 0 to 783
	generate
	
		lfsr	: lfsr_13bit 

			port map(
				-- input
				clk		=> clk,
				en		=> en,
				load_n		=> load_n,
				lfsr_in		=> lfsr_in(i),

				-- output
				lfsr_out	=> lfsr_out(i)
			);

	end generate lfsr_layer;


	cmp_layer	: for i in 0 to 783
	generate
	
		cmp	: unsigned_cmp_gt 
		
			generic map(
				N		=> 8
			)


			port map(
				-- input
				in0		=> unsigned(lfsr_out(i)),
				in1		=> unsigned(pixels_sig(i)),

				-- output		
				cmp_out		=> spikes(i)
			);

	end generate cmp_layer;



end architecture behaviour;
