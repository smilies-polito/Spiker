library ieee;
use ieee.std_logic_1164.all;

entity bitMask is

	generic(
		N_cnt		: integer :=3
	);

	port(
		-- input
		input_cnt	: in std_logic_vector(N_cnt-1 downto 0);
		inh		: in std_logic;
		input_bit	: in std_logic;

		-- output
		output_bits	: out std_logic_vector(2**N_cnt-1 downto 0)
	);

end entity bitMask;


architecture behaviour of bitMask is

	signal inhibited_sel	: std_logic_vector(2**N_cnt-1 downto 0);
	signal neuron_sel	: std_logic_vector(2**N_cnt-1 downto 0);


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

	inh_and_computation		: process(inh, neuron_sel)
	begin
		for i in 0 to 2**N_cnt-1
		loop
			inhibited_sel(i)	<= inh and neuron_sel(i);
		end loop;
	end process inh_and_computation;


	input_bit_and_computation	: process(input_bit, inhibited_sel)
	begin
		for i in 0 to 2**N_cnt-1
		loop
			output_bits(i)	<= input_bit and not inhibited_sel(i);
		end loop;
	end process input_bit_and_computation;



	cnt_decoder	: decoder
		generic map(
			N		=> N_cnt
		)

		port map(
			-- input
			encoded_in	=> input_cnt,

			-- output
			decoded_out	=> neuron_sel
		);

end architecture behaviour;
