library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_cu_dp_tb is
end entity neuron_cu_dp_tb;



architecture behaviour of neuron_cu_dp_tb is

	-- parallelism
	constant N		: integer := 8;

	-- exponential shift
	constant shift		: integer := 1;

	-- model parameters
	constant v_th_0_int	: integer := -52;	
	constant v_rest_int	: integer := -65;	
	constant v_reset_int	: integer := -60;	
	constant v_th_plus_int	: integer := 1;	
	constant inh_weight_int	: integer := -15;	
	constant exc_weight_int	: integer := 3;	


	-- input controls
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal exp_exc_start	: std_logic;
	signal rest_inh_start	: std_logic;
	signal mask_neuron	: std_logic;
	signal input_spike	: std_logic;

	-- input parameters
	signal v_th_0		: signed(N-1 downto 0);
	signal v_rest		: signed(N-1 downto 0);
	signal v_reset		: signed(N-1 downto 0);
	signal v_th_plus	: signed(N-1 downto 0);
	signal inh_weight	: signed(N-1 downto 0);
	signal exc_weight	: signed(N-1 downto 0);

	-- from control unit towards datapath
	signal v_update_sel	: std_logic_vector(1 downto 0);
	signal v_or_v_th	: std_logic;
	signal add_or_sub	: std_logic;
	signal update_or_rest	: std_logic;
	signal v_reset_sel	: std_logic;
	signal v_th_en		: std_logic;
	signal v_en		: std_logic;
	
	-- from datapath towards control unit
	signal exceed_v_th	: std_logic;

	-- datapath internal signals
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


	-- output
	signal out_spike	: std_logic;


	type states is (
		reset,
		idle,
		exp_decay1,
		exp_decay2,
		fire,
		pause,
		exc_spike,
		no_exc_spike,
		inh_spike,
		no_inh_spike,
		rest
	);

	signal present_state, next_state	: states;
	signal masked_spike			: std_logic;




	-- datapath components
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


	-- clock
	clock_gen : process
	begin
		clk	<= '0';			-- falling edge i*12ns
		wait for 6 ns;			                            
		clk	<= '1';                 -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- reset
	reset_gen : process
	begin
		rst_n	<= '1';
		wait for 14 ns;
		rst_n	<= '0';
		wait for 3 ns;
		rst_n	<= '1';
		wait;
	end process reset_gen;



	-- exp_exc_start
	exp_exc_start_gen : process
	begin
		exp_exc_start	<= '0';		-- 0 ns
		wait for 26 ns;			
		exp_exc_start	<= '1';		-- 26 ns
		wait for 12 ns;			          
		exp_exc_start	<= '0';         -- 38 ns
		wait for 12 ns;			          
		exp_exc_start	<= '1';         -- 50 ns
		wait for 60 ns;			          
		exp_exc_start	<= '0';         -- 110 ns
		wait for 60 ns;
		exp_exc_start	<= '1';         -- 170 ns
		wait for 12 ns;
		exp_exc_start	<= '0';         -- 182 ns
		wait for 12 ns;
		exp_exc_start	<= '1';         -- 194 ns
		wait for 36 ns;
		exp_exc_start	<= '0';         -- 230 ns
		wait for 24 ns;
		exp_exc_start	<= '1';		-- 254 ns
		wait for 12 ns;
		exp_exc_start	<= '0';         -- 266 ns
		wait for 36 ns;
		exp_exc_start	<= '1';		-- 302 ns
		wait for 12 ns;
		exp_exc_start	<= '0';         -- 314 ns
		wait;
	end process exp_exc_start_gen;


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';		-- 0 ns	
		wait for 50 ns;			
		input_spike	<= '1';		-- 50 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 62 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 74 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 86 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 98 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 110 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 122 ns
		wait for 24 ns;			          
		input_spike	<= '0';         -- 146 ns
		wait for 48 ns;
		input_spike	<= '1';         -- 194 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 206 ns
		wait for 12 ns;			          
		input_spike	<= '1';         -- 218 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 230 ns	
		wait for 48 ns;
		input_spike	<= '1';		-- 278 ns
		wait for 12 ns;			          
		input_spike	<= '0';         -- 290 ns	
		wait;
	end process input_spike_gen;


	-- rest_inh_start
	rest_inh_start_gen : process
	begin
		rest_inh_start	<= '0';		-- 0 ns
		wait for 110 ns;		
		rest_inh_start	<= '1';		-- 110 ns
		wait for 36 ns;			          
		rest_inh_start	<= '0';         -- 146 ns
		wait for 132 ns;		
		rest_inh_start	<= '1';		-- 278 ns
		wait for 12 ns;
		rest_inh_start	<= '0';		-- 290 ns
		wait for 48 ns;
		rest_inh_start	<= '1';		-- 338 ns
		wait for 12 ns;
		rest_inh_start	<= '0';		-- 350 ns
		wait;
	end process rest_inh_start_gen;



	-- mask_neuron
	mask_neuron_gen : process
	begin
		mask_neuron	<= '0';		-- 0 ns
		wait for 134 ns;
		mask_neuron	<= '1';		-- 134 ns
		wait for 12 ns;
		mask_neuron	<= '0';		-- 146 ns
		wait for 48 ns;
		mask_neuron	<= '1';		-- 194 ns
		wait;
	end process mask_neuron_gen;



	-- model parameters binary conversion
	v_th_0		<= to_signed(v_th_0_int, N);
	v_rest		<= to_signed(v_rest_int, N);
	v_reset		<= to_signed(v_reset_int, N);
	v_th_plus	<= to_signed(v_th_plus_int, N);
	inh_weight	<= to_signed(inh_weight_int, N);
	exc_weight	<= to_signed(exc_weight_int, N);







	-- control unit
	masked_spike <= input_spike and not mask_neuron;
	
	

	-- state transition
	state_transition	: process(clk, rst_n)
	begin

		-- active reset
		if rst_n = '0'
		then
			present_state <= reset;

		-- inactive reset
		elsif clk'event and clk = '1'
		then
			present_state <= next_state;
		end if;

	end process state_transition;




	-- state evaluation
	state_evaluation	: process(present_state, exp_exc_start,
					exceed_v_th, rest_inh_start, 
					input_spike, masked_spike)
	begin

		-- default case
		next_state	<= reset;

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>
				if exp_exc_start = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;
					else
						next_state <= exp_decay1;
					end if;
				else
					if rest_inh_start = '1'
					then
						next_state <= rest;
					else
						next_state <= idle;
					end if;
				end if;

			-- exp_decay1
			when exp_decay1 =>
				next_state <= exp_decay2;
				
			-- exp_decay2
			when exp_decay2 =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- fire
			when fire =>
				next_state <= pause;

			-- pause
			when pause =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- exc_spike
			when exc_spike =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- no_exc_spike
			when no_exc_spike =>
				if exp_exc_start = '1'
				then
					if input_spike = '1'
					then
						next_state <= exc_spike;
					else
						next_state <= no_exc_spike;
					end if;
				else
					if rest_inh_start = '1'
					then
						if input_spike = '1'
						then
							next_state <= inh_spike;
						else
							next_state <= no_inh_spike;
						end if;
					else
						next_state <= idle;
					end if;
				end if;

			-- inh_spike
			when inh_spike =>
				if rest_inh_start = '1'
				then
					if masked_spike = '1'
					then
						next_state <= inh_spike;
					else
						next_state <= no_inh_spike;
					end if;
				else
					next_state <= idle;
				end if;

			-- no_inh_spike
			when no_inh_spike =>
				if rest_inh_start = '1'
				then
					if masked_spike = '1'
					then
						next_state <= inh_spike;
					else
						next_state <= no_inh_spike;
					end if;
				else
					next_state <= idle;
				end if;

			-- rest
			when rest =>
				next_state <= idle;
			-- default case
			when others =>
				next_state <= reset;

		end case;

	end process state_evaluation;





	output_evaluation	: process(present_state)
	begin

		-- default values
		v_update_sel	<= "00";
		v_or_v_th	<= '0';
		add_or_sub	<= '0';
		update_or_rest	<= '1';
		v_reset_sel	<= '0';
		v_th_en		<= '0';
		v_en		<= '0';
		out_spike	<= '0';

		case present_state is

			-- reset
			when reset =>
				v_en 		<= '1';
				v_th_en		<= '1';
				v_reset_sel	<= '1';
				update_or_rest	<= '0';

			-- idle
			when idle =>
				v_en		<= '0';
				v_th_en		<= '0';

			-- exp_decay1
			when exp_decay1 =>
				v_update_sel	<= "00";
				add_or_sub	<= '1';
				v_en		<= '1';

			-- exp_decay2
			when exp_decay2	=>
				v_update_sel	<= "01";
				add_or_sub	<= '0';
				v_en		<= '1';
				
			-- fire
			when fire =>
				v_or_v_th	<= '1';
				add_or_sub	<= '0';
				v_reset_sel	<= '1';
				v_th_en		<= '1';
				v_en		<= '1';
				out_spike	<= '1';

			-- pause
			when pause =>
				v_en		<= '0';
				v_th_en		<= '0';

			-- exc_spike
			when exc_spike =>
				v_update_sel	<= "10";
				add_or_sub	<= '0';
				v_en		<= '1';

			-- no_exc_spike
			when no_exc_spike	=>
				v_update_sel	<= "10";
				add_or_sub	<= '0';
				v_en		<= '0';
				
			-- inh_spike
			when inh_spike		=>
				v_update_sel	<= "11";
				add_or_sub	<= '0';
				v_en		<= '1';

			-- no_inh_spike
			when no_inh_spike	=>
				v_update_sel	<= "11";
				add_or_sub	<= '0';
				v_en		<= '0';

			-- rest
			when rest =>
				update_or_rest	<= '0';
				v_en		<= '1';

			-- default case
			when others =>
				v_update_sel	<= "00";
				v_or_v_th	<= '0';
				add_or_sub	<= '0';
				update_or_rest	<= '1';
				v_reset_sel	<= '0';
				v_th_en		<= '0';
				v_en		<= '0';
				out_spike	<= '0';

		end case;

	end process output_evaluation;





	-- datapath
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


end architecture behaviour;
