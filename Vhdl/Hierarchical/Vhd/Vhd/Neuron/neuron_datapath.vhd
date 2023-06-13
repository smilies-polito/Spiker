library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_datapath is

	generic(
		-- parallelism
		neuron_bit_width		: integer := 16;
		weights_bit_width		: integer := 5;

		-- shift amount
		shift				: integer := 1
	);

	port(
		-- input parameters
		v_th_value		: in signed(neuron_bit_width-1 
						downto 0);
		v_reset			: in signed(neuron_bit_width-1 
						downto 0);
		inh_weight		: in signed(weights_bit_width-1
						downto 0);
		exc_weight		: in signed(weights_bit_width-1
						downto 0);

		-- input controls
		clk			: in std_logic;
		update_sel		: in std_logic_vector(1 downto 0);
		add_or_sub		: in std_logic;
		v_update		: in std_logic;
		v_th_en			: in std_logic;
		v_en			: in std_logic;
		v_rst_n			: in std_logic;
		
		-- output
		exceed_v_th		: out std_logic		
	);

end entity neuron_datapath;


architecture behaviour of neuron_datapath is

	
	signal update		: signed(neuron_bit_width-1 downto 0);
	signal update_value	: signed(neuron_bit_width-1 downto 0);
	signal v_th		: signed(neuron_bit_width-1 downto 0);
	signal v_value		: signed(neuron_bit_width-1 downto 0);
	signal v		: signed(neuron_bit_width-1 downto 0);
	signal v_shifted	: signed(neuron_bit_width-1 downto 0);



	component shifter is

		generic(
			-- parallelism
			N		: integer := 8;
		
			-- shift
			shift		: integer := 1	
		);

		port(
			-- input
			shifter_in	: in signed(N-1 downto 0);

			-- output
			shifted_out	: out signed(N-1 downto 0)
		);

	end component shifter;


	component mux4to1_signed is

		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			sel	: in std_logic_vector(1 downto 0);
			in0	: in signed(N-1 downto 0);
			in1	: in signed(N-1 downto 0);
			in2	: in signed(N-1 downto 0);
			in3	: in signed(N-1 downto 0);

			-- output
			mux_out	: out signed(N-1 downto 0)
		);

	end component mux4to1_signed;


	component add_sub is

		generic(
			N		: integer := 8		
		);

		port(
			-- input
			in0		: in signed(N-1 downto 0);
			in1		: in signed(N-1 downto 0);
			add_or_sub	: in std_logic;

			-- output
			add_sub_out	: out signed(N-1 downto 0)		
		);

	end component add_sub;


	component mux2to1_signed is

		generic(
			-- parallelism
			N	: integer		
		);

		port(	
			-- inputs	
			sel	: in std_logic;
			in0	: in signed(N-1 downto 0);
			in1	: in signed(N-1 downto 0);

			-- output
			mux_out	: out signed(N-1 downto 0)
		);

	end component mux2to1_signed;


	
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



	component reg_signed_sync_rst is

		generic(
			-- parallelism
			N	: integer	:= 16		
		);

		port(	
			-- inputs	
			clk	: in std_logic;
			en	: in std_logic;
			rst_n	: in std_logic;
			reg_in	: in signed(N-1 downto 0);

			-- outputs
			reg_out	: out signed(N-1 downto 0)
		);

	end component reg_signed_sync_rst;



	component cmp_gt is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			in0	: in signed(N-1 downto 0);
			in1	: in signed(N-1 downto 0);

			-- output
			cmp_out	: out std_logic
		);

	end component cmp_gt;



begin

	v_shifter	: shifter
		generic map(
			-- parallelism
			N		=> neuron_bit_width,
		
			-- shift
			shift		=> shift
		)

		port map(
			-- input
			shifter_in	=> v,

			-- output
			shifted_out	=> v_shifted
		);

	
	update_mux	: mux4to1_signed
		generic map(
			-- parallelism
			N	=> neuron_bit_width
		)
		port map(
			-- input
			sel				=> update_sel,
			in0				=> (others => '0'),

			in1				=> v_shifted,

			in2(neuron_bit_width-1 
			downto 
			weights_bit_width)		=> (others =>
							exc_weight(
							weights_bit_width - 1)),
	
			in2(weights_bit_width-1 
			downto 0)			=> exc_weight,

			in3(neuron_bit_width-1 
			downto 
			weights_bit_width)		=> (others =>
							inh_weight(
							weights_bit_width - 1)),
			in3(weights_bit_width-1 
			downto 0)			=> inh_weight,

			-- output
			mux_out	=> update
		);


	update_add_sub	: add_sub
		generic map(
			N		=> neuron_bit_width
		)

		port map(
			-- input
			in0		=> v,
			in1		=> update,
			add_or_sub	=> add_or_sub,

			-- output
			add_sub_out	=> update_value
		);


	v_mux	: mux2to1_signed
		generic map(
			-- parallelism
			N	=> neuron_bit_width
		)

		port map(	
			-- inputs	
			sel	=> v_update,
			in0	=> update_value,
			in1	=> v_reset,

			-- outpu
			mux_out	=> v_value
		);



	v_th_reg	: reg_signed
		generic map(
			-- parallelism
			N	=> neuron_bit_width
		)

		port map(	
			-- input
			clk	=> clk,	
			en	=> v_th_en,
			reg_in	=> v_th_value,

			-- output
			reg_out	=> v_th
		);



	v_reg		: reg_signed_sync_rst
		generic map(
			-- parallelism
			N	=> neuron_bit_width
		)

		port map(	
			-- input
			clk	=> clk,	
			en	=> v_en,
			rst_n	=> v_rst_n,
			reg_in	=> v_value,

			-- output
			reg_out	=> v
		);


	fire_cmp	: cmp_gt
		generic map(
			N	=> neuron_bit_width
		)

		port map(
			-- input
			in0	=> v, 
			in1	=> v_th,

			-- output
			cmp_out	=> exceed_v_th
		);

end architecture behaviour;
