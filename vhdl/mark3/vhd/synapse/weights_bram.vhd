library ieee;
use ieee.std_logic_1164.all;



entity weights_bram is

	port(
		-- input
		clk		: in std_logic;
		di		: in std_logic_vector(71 downto 0);
		rdaddr		: in std_logic_vector(15 downto 0);
		rden		: in std_logic;
		wraddr		: in std_logic_vector(15 downto 0);
		bram_sel	: in std_logic_vector(5 downto 0);

		-- output
		do		: out std_logic_vector(400*5-1 downto 0)
				
	);

end entity weights_bram;



architecture behaviour of weights_bram is

	type data_matrix is array(44 downto 0) of std_logic_vector(71 downto 0);

	signal wren	: std_logic_vector(63 downto 0);
	signal data_out	: data_matrix;
	

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


	component bram_sdp is

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(71 downto 0);
			rdaddr		: in std_logic_vector(15 downto 0);
			rden		: in std_logic;
			wraddr		: in std_logic_vector(15 downto 0);
			wren		: in std_logic;

			-- output
			do		: out std_logic_vector(71 downto 0)
					
		);

	end component bram_sdp;


begin


	bram_decoder	: decoder
		generic map(
			N		=> 6
		)

		port map(
			-- input
			encoded_in	=> bram_sel,

			-- output
			decoded_out	=> wren
		);

	complete_memory	: for i in 0 to 44
	generate

		single_bram	:  bram_sdp 
			port map(
				-- input
				clk		=> clk,
				di		=> di,
				rdaddr		=> rdaddr,
				rden		=> rden,
				wraddr		=> wraddr,
				wren		=> wren(i),

				-- output
				do		=> data_out(i)
						
			);


	end generate complete_memory;


	connect_output	: process(data_out)
	begin

		for i in 0 to 43
		loop

			do((i+1)*9*5-1 downto i*9*5) <= data_out(i)(9*5-1 
				downto 0);

		end loop;

		do(400*5-1 downto 44*9*5) <= data_out(44)(4*5-1 downto 0);

	end process connect_output;


end architecture behaviour;
