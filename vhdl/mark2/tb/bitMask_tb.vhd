library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bitMask_tb is
end entity bitMask_tb;


architecture behaviour of bitMask_tb is


	constant N_cnt		: integer :=3;

	-- input
	signal input_cnt	: std_logic_vector(N_cnt-1 downto 0);
	signal inh		: std_logic;
	signal input_bit	: std_logic;

	-- output
	signal output_bits	: std_logic_vector(2**N_cnt-1 downto 0);
	

	-- DUT internal signals
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


	input_cnt_gen	: process
	begin

		for i in 0 to 2
		loop
			for i in 0 to 2**N_cnt-1
			loop
				input_cnt	<= std_logic_vector(to_unsigned(i, N_cnt));
				wait for 10 ns;
			end loop;
		
			wait for 50 ns;
		end loop;

	end process input_cnt_gen;


	inh_gen		: process
	begin

		inh		<= '0';
		wait for 100 ns;
		inh		<= '1';
		wait;

	end process inh_gen;


	input_bit_gen	: process
	begin

		input_bit	<= '1';
		wait;

	end process input_bit_gen;


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
