library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_datapath_tb is
end entity neuron_datapath_tb;


architecture test of neuron_datapath_tb is


	
	-- parallelism
	constant N		: integer := 8;

	-- exponential shift
	constant shift		: integer := 1;

	-- model parameters
	constant v_th_value_int	: integer := 13;	
	constant v_reset_int	: integer := 5;	
	constant inh_weight_int	: integer := -15;	
	constant exc_weight_int	: integer := 3;	

	-- input parameters
	signal v_th_value	: signed(N-1 downto 0);
	signal v_reset		: signed(N-1 downto 0);
	signal inh_weight	: signed(N-1 downto 0);
	signal exc_weight	: signed(N-1 downto 0);

	-- input controls
	signal clk		: std_logic;
	signal update_sel	: std_logic_vector(1 downto 0);
	signal add_or_sub	: std_logic;
	signal v_update		: std_logic;
	signal v_th_en		: std_logic;
	signal v_en		: std_logic;
	signal v_rst_n		: std_logic;
	
	-- output
	signal exceed_v_th	: std_logic;

	signal update		: signed(N-1 downto 0);
	signal update_value	: signed(N-1 downto 0);
	signal v_th		: signed(N-1 downto 0);
	signal v_value		: signed(N-1 downto 0);
	signal v		: signed(N-1 downto 0);
	signal v_shifted	: signed(N-1 downto 0);


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

	-- model parameters binary conversion
	v_th_value	<= to_signed(v_th_value_int, N);
	v_reset		<= to_signed(v_reset_int, N);
	inh_weight	<= to_signed(inh_weight_int, N);
	exc_weight	<= to_signed(exc_weight_int, N);



	clock			: process
	begin
		clk		<= '0';
		wait for 6 ns;
		clk		<= '1';
		wait for 6 ns;
	end process clock;



	update_sel_process	: process
	begin

		update_sel	<= "00";	-- 0 ns
		wait for 20 ns;
		update_sel	<= "01";	-- 20 ns
		wait for 10 ns;
		update_sel	<= "10";	-- 30 ns
		wait for 10 ns;
		update_sel	<= "11";	-- 40 ns
		wait for 110 ns;
		update_sel	<= "10";	-- 150 ns
		wait;

	end process update_sel_process;



	add_or_sub_process	: process
	begin

		add_or_sub	<= '0';		-- 0 ns
		wait for 60 ns;
		add_or_sub	<= '1';		-- 60 ns
		wait;

	end process add_or_sub_process;


	v_update_process	: process
	begin

		v_update	<= '0';		-- 0 ns
		wait for 70 ns;
		v_update	<= '1';		-- 70 ns
		wait;

	end process v_update_process;


	v_th_en_process	: process
	begin

		v_th_en	<= '1';			-- 0 ns
		wait for 10 ns;
		v_th_en	<= '0';			-- 10 ns
		wait;

	end process v_th_en_process;


	v_en_process	: process
	begin

		v_en	<= '1';			-- 0 ns
		wait for 10 ns;
		v_en	<= '0';			-- 10 ns
		wait for 90 ns;
		v_en	<= '1';			-- 100 ns
		wait;

	end process v_en_process;


	v_rst_n_process	: process
	begin

		v_rst_n	<= '1';			-- 0 ns
		wait for 110 ns;
		v_rst_n	<= '0';			-- 110 ns
		wait for 10 ns;
		v_rst_n	<= '1';			-- 120 ns
		wait;

	end process v_rst_n_process;





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

	
	update_mux	: mux4to1_signed
		generic map(
			-- parallelism
			N	=> N		
		)
		port map(
			-- input
			sel	=> update_sel,
			in0	=> (others => '0'),
			in1	=> v_shifted,
			in2	=> exc_weight,
			in3	=> inh_weight,
		                               
			-- output
			mux_out	=> update
		);


	update_add_sub	: add_sub
		generic map(
			N		=> N	
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
			N	=> N
		)

		port map(	
			-- inputs	
			sel	=> v_update,
			in0	=> v_reset,
			in1	=> update_value,

			-- outpu
			mux_out	=> v_value
		);



	v_th_reg	: reg_signed
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



	v_reg		: reg_signed_sync_rst
		generic map(
			-- parallelism
			N	=> N   
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
			N	=> N	
		)

		port map(
			-- input
			in0	=> v_value, 
			in1	=> v_th,

			-- outpu
			cmp_out	=> exceed_v_th
		);



end architecture test;
