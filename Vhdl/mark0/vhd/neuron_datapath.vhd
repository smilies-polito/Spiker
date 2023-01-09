library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_datapath is

	generic(
		-- parallelism
		N			: integer := 8;

		-- shift amount
		shift			: integer := 1
	);

	port(
		-- input parameters
		v_th_0		: in signed(N-1 downto 0);
		v_rest		: in signed(N-1 downto 0);
		v_reset		: in signed(N-1 downto 0);
		v_th_plus	: in signed(N-1 downto 0);
		inh_weight	: in signed(N-1 downto 0);
		exc_weight	: in signed(N-1 downto 0);

		-- input controls
		clk		: in std_logic;
		v_update_sel	: in std_logic_vector(1 downto 0);
		v_or_v_th	: in std_logic;
		add_or_sub	: in std_logic;
		update_or_rest	: in std_logic;
		v_reset_sel	: in std_logic;
		v_th_en		: in std_logic;
		v_en		: in std_logic;
		
		-- output
		exceed_v_th	: out std_logic		
	);

end entity neuron_datapath;


architecture test of neuron_datapath is

	
	signal v_update		: signed(N-1 downto 0);
	signal update		: signed(N-1 downto 0);
	signal prev_value	: signed(N-1 downto 0);
	signal update_value	: signed(N-1 downto 0);
	signal v_th_value	: signed(N-1 downto 0);
	signal v_th		: signed(N-1 downto 0);
	signal v_next		: signed(N-1 downto 0);
	signal v_value		: signed(N-1 downto 0);
	signal v		: signed(N-1 downto 0);
	signal v_shifted	: signed(N-1 downto 0);
	signal v_rest_shifted	: signed(N-1 downto 0);


	component mux2to1 is

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

	end component mux2to1;


	component mux4to1 is

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

	end component mux4to1;


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


	component reg is

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

	end component reg;


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


	v_rest_shifter	: shifter
		generic map(
			-- parallelism
			N		=> N,
		
			-- shift
			shift		=> shift
		)

		port map(
			-- input
			shifter_in	=> v_rest,

			-- output
			shifted_out	=> v_rest_shifted
		);


	v_shifter	: shifter
		generic map(
			-- parallelism
			N		=> N,
		
			-- shift
			shift		=> shift
		)

		port map(
			-- input
			shifter_in	=> v,

			-- output
			shifted_out	=> v_shifted
		);


	v_update_mux	: mux4to1
		generic map(
			-- parallelism
			N	=> N		
		)
		port map(
			-- input
			sel	=> v_update_sel,
			in0	=> v_shifted,
			in1	=> v_rest_shifted,
			in2	=> exc_weight,
			in3	=> inh_weight,
		                               
			-- output
			mux_out	=> v_update
		);


	update_mux	: mux2to1
		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> v_or_v_th,
			in0	=> v_update,
			in1	=> v_th_plus,

			-- outpu
			mux_out	=> update
		);


	prev_mux	: mux2to1	
		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> v_or_v_th,
			in0	=> v,
			in1	=> v_th,

			-- outpu
			mux_out	=> prev_value
		);

	
	update_add_sub	: add_sub
		generic map(
			N		=> N	
		)

		port map(
			-- input
			in0		=> prev_value,
			in1		=> update,
			add_or_sub	=> add_or_sub,

			-- output
			add_sub_out	=> update_value
		);


	v_th_mux	: mux2to1
		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> update_or_rest,
			in0	=> v_th_0,
			in1	=> update_value,

			-- outpu
			mux_out	=> v_th_value
		);

	v_mux	: mux2to1
		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> update_or_rest,
			in0	=> v_rest,
			in1	=> update_value,

			-- outpu
			mux_out	=> v_next
		);


	reset_mux	: mux2to1
		generic map(
			-- parallelism
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> v_reset_sel,
			in0	=> v_next,
			in1	=> v_reset,

			-- outpu
			mux_out	=> v_value
		);


	v_th_reg	: reg
		generic map(
			-- parallelism
			N	=> N   
		)

		port map(	
			-- input
			clk	=> clk,	
			en	=> v_th_en,
			reg_in	=> v_th_value,

			-- output
			reg_out	=> v_th
		);



	v_reg		: reg
		generic map(
			-- parallelism
			N	=> N   
		)

		port map(	
			-- input
			clk	=> clk,	
			en	=> v_en,
			reg_in	=> v_value,

			-- output
			reg_out	=> v
		);


	fire_cmp	: cmp_gt
		generic map(
			N	=> N	
		)

		port map(
			-- input
			in0	=> v, 
			in1	=> v_th,

			-- outpu
			cmp_out	=> exceed_v_th
		);

	

end architecture test;
