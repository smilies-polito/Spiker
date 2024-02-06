library ieee;
use ieee.std_logic_1164.all;
use ieee.math_real."ceil";
use ieee.math_real."log2";

entity multi_input_dp is

	generic(
		n_exc		: integer	:= 15;
		n_exc_bit_width	: integer	:= 4;
 		n_inh		: integer	:= 8;	
 		n_inh_bit_width	: integer	:= 3	
	);

	port(
	
    		-- input from the outside world
		clk		: in std_logic;
		n_exc_reg_en	: in std_logic;
		n_exc_value	: in std_logic_vector(n_exc_bit_width-1 
					downto 0);
		n_inh_reg_en	: in std_logic;
		n_inh_value	: in std_logic_vector(n_inh_bit_width-1
					downto 0);
		exc_spikes	: in std_logic_vector(n_exc-1 downto 0);
		inh_spikes	: in std_logic_vector(n_inh-1 downto 0);

		-- input from control unit
		exc_cnt_en	: in std_logic;
		exc_cnt_rst_n	: in std_logic;
		inh_cnt_en	: in std_logic;
		inh_cnt_rst_n	: in std_logic;

		-- output towards control unit
		exc_yes		: out std_logic;
		exc_stop	: out std_logic;
		inh_yes		: out std_logic;
		inh_stop	: out std_logic;

		-- output towards the outside world
		exc_spike	: out std_logic;
		exc_cnt		: out std_logic_vector(n_exc_bit_width-1
					downto 0);
		inh_spike	: out std_logic;
		inh_cnt		: out std_logic_vector(n_inh_bit_width-1
					downto 0)
	);

end entity multi_input_dp;


architecture behavior of multi_input_dp is

	signal exc_cnt_sig	: std_logic_vector(n_exc_bit_width-1 downto 0);
	signal inh_cnt_sig	: std_logic_vector(n_inh_bit_width-1 downto 0);

	signal exc_spikes_samp	: std_logic_vector(n_exc-1 downto 0);
	signal inh_spikes_samp	: std_logic_vector(n_inh-1 downto 0);

	signal n_exc_tc		: std_logic_vector(n_exc-1 downto 0);
	signal n_inh_tc		: std_logic_vector(n_inh-1 downto 0);

	signal exc_mux_in	: std_logic_vector(2**n_exc_bit_width-1 
					downto 0);
	signal inh_mux_in	: std_logic_vector(2**n_inh_bit_width-1 
					downto 0);

	component generic_or is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			or_in	: in std_logic_vector(N-1 downto 0);

			-- output
			or_out	: out std_logic
		);

	end component generic_or;


	component generic_mux_1bit is

		generic(
			N_sel	: integer := 8		
		);

		port(
			-- input
			mux_in	: in std_logic_vector(2**N_sel-1 downto 0);
			mux_sel	: in std_logic_vector(N_sel-1 downto 0);

			-- output
			mux_out	: out std_logic
		);

	end component generic_mux_1bit;


	component cnt is

		generic(
			bit_width	: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(bit_width-1 
						downto 0)
		);

	end component cnt;


	component cmp_eq is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			in0	: in std_logic_vector(N-1 downto 0);
			in1	: in std_logic_vector(N-1 downto 0);

			-- output
			cmp_out	: out std_logic
		);

	end component cmp_eq;

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

begin

	-- zero padding of the input to fill all the multiplexer's inputs
	exc_mux_in(n_exc-1 downto 0) 			<= exc_spikes;
	exc_mux_in(2**n_exc_bit_width-1 downto n_exc) 	<= (others => '0');

	inh_mux_in(n_inh-1 downto 0) 			<= inh_spikes;
	inh_mux_in(2**n_inh_bit_width-1 downto n_inh) 	<= (others => '0');

	exc_cnt	<= exc_cnt_sig;
	inh_cnt	<= inh_cnt_sig;

	inputs_or	: generic_or

		generic map(
			N		=> n_exc
		)

		port map (
			-- input
			or_in		=> exc_spikes,

			-- output
			or_out		=> exc_yes
		);



	inputs_mux	: generic_mux_1bit

		generic map(
			N_sel		=> n_exc_bit_width	
		)

		port map(
			-- input
			mux_in		=> exc_mux_in,
			mux_sel		=> exc_cnt_sig,

			-- output
			mux_out		=> exc_spike
		);


	select_cnt	: cnt

		generic map(
			bit_width	=> n_exc_bit_width
		)

		port map(
			-- input
			clk		=> clk,		
			cnt_en		=> exc_cnt_en,
			cnt_rst_n	=> exc_cnt_rst_n,

			-- output
			cnt_out		=> exc_cnt_sig
		);


--	end_cmp		: cmp_eq
--
--		generic map(
--			N		=> N_cnt
--		)
--
--		port map(
--			-- input
--			in0		=> cnt_out,
--			in1		=> N_inputs,
--
--			-- outpu
--			cmp_out		=> stop
--		);


end architecture behavior;
