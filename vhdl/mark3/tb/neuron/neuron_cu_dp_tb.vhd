library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_cu_dp_tb is
end entity neuron_cu_dp_tb;



architecture behaviour of neuron_cu_dp_tb is


	-- parallelism
	constant N		: integer := 16;
	constant N_weight	: integer := 15;

	-- exponential shift
	constant shift		: integer := 10;


	-- model parameters
	constant v_th_value_int	: integer 	:= 13*(2**10);	
	constant v_reset_int	: integer 	:= 5*(2**10);	
	constant v_th_plus_int	: integer	:= 102; -- 0.1*2^11 rounded	
	constant inh_weight_int	: integer 	:= 7*(2**10);	
	constant exc_weight_int	: integer 	:= 7*(2**10);

	-- input parameters
	signal v_th_value	: signed(N-1 downto 0);
	signal v_reset		: signed(N-1 downto 0);
	signal inh_weight	: signed(N-1 downto 0);
	signal exc_weight	: signed(N_weight-1 downto 0);


	-- input
	signal clk		: std_logic;
	signal rst_n		: std_logic;
	signal start		: std_logic;
	signal stop		: std_logic;
	signal exc_or		: std_logic;
	signal exc_stop		: std_logic;
	signal inh_or		: std_logic;
	signal inh_stop		: std_logic;
        signal input_spike	: std_logic;

	-- input for the initial load
	signal v_th_en		: std_logic;

	-- from datapath to cu
	signal exceed_v_th	: std_logic;

	-- from cu to datapath
	signal update_sel	: std_logic_vector(1 downto 0);
	signal add_or_sub	: std_logic;
	signal v_update		: std_logic;
	signal v_en		: std_logic;
	signal v_rst_n		: std_logic;

	-- output
	signal out_spike	: std_logic;
	signal neuron_ready	: std_logic;


	-- datapath internal signals
	signal update		: signed(N-1 downto 0);
	signal update_value	: signed(N-1 downto 0);
	signal v_th		: signed(N-1 downto 0);
	signal v_value		: signed(N-1 downto 0);
	signal v		: signed(N-1 downto 0);
	signal v_shifted	: signed(N-1 downto 0);


	-- control unit internal signals
	type states is (
		reset,
		idle,
		exp_decay,
		no_exc_spike,
		exc_spike,
		no_inh_spike,
		inh_spike,
		fire
	);


	signal present_state, next_state	: states;


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


	-- model parameters binary conversion
	v_th_value	<= to_signed(v_th_value_int, N);
	v_reset		<= to_signed(v_reset_int, N);
	inh_weight	<= to_signed(inh_weight_int, N);
	exc_weight	<= to_signed(exc_weight_int, N_weight);



	-- clock
	clock_gen : process
	begin
		clk	<= '0';		-- falling edge i*12ns
		wait for 6 ns;			                    
		clk	<= '1';         -- rising edge 6ns + i*12ns
		wait for 6 ns;			
	end process clock_gen;



	-- reset
	reset_gen : process
	begin
		rst_n	<= '1';		-- 0 ns
		wait for 14 ns;
		rst_n	<= '0';		-- 14 ns
		wait for 3 ns;
		rst_n	<= '1';		-- 17 ns
		wait;
	end process reset_gen;



	-- v_th_en
	v_th_en_gen : process
	begin
		v_th_en	<= '0';		-- 0 ns
		wait for 26 ns;
		v_th_en	<= '1';		-- 26 ns
		wait for 12 ns;
		v_th_en	<= '0';		-- 38 ns
		wait;
	end process v_th_en_gen;




	-- start
	start_gen : process
	begin
		start	<= '0';		-- 0 ns
		wait for 38 ns;
		start	<= '1';		-- 38 ns
		wait for 12 ns;
		start	<= '0';		-- 50 ns
		wait;
	end process start_gen;


	-- exc_or
	exc_or_gen : process
	begin
		exc_or	<= '0';		-- 0 ns
		wait for 62 ns;			
		exc_or	<= '1';         -- 62 ns
		wait for 48 ns;			          
		exc_or	<= '0';         -- 110 ns
		wait for 48 ns;			          
		exc_or	<= '1';         -- 158 ns
		wait for 84 ns;			          
		exc_or	<= '0';         -- 242 ns
		wait;
	end process exc_or_gen;


	-- exc_stop
	exc_stop_gen : process
	begin
		exc_stop	<= '0';		-- 0 ns
		wait for 98 ns;			
		exc_stop	<= '1';         -- 98 ns
		wait for 12 ns;			          
		exc_stop	<= '0';         -- 110 ns
		wait for 84 ns;			          
		exc_stop	<= '1';         -- 194 ns
		wait for 12 ns;			          
		exc_stop	<= '0';         -- 206 ns
		wait;
	end process exc_stop_gen;


	-- inh_or
	inh_or_gen : process
	begin
		inh_or	<= '0';		-- 0 ns
		wait for 110 ns;			          
		inh_or	<= '1';         -- 110 ns
		wait for 132 ns;			          
		inh_or	<= '0';         -- 242 ns
		wait;
	end process inh_or_gen;


	-- inh_stop
	inh_stop_gen : process
	begin
		inh_stop	<= '0';		-- 0 ns
		wait for 146 ns;			
		inh_stop	<= '1';         -- 146 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 158 ns
		wait for 72 ns;			          
		inh_stop	<= '1';         -- 230 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 242 ns
		wait for 12 ns;
		inh_stop	<= '1';		-- 254 ns
		wait for 12 ns;			          
		inh_stop	<= '0';         -- 266 ns
		wait;
	end process inh_stop_gen;


	-- input_spike
	input_spike_gen: process
	begin
		input_spike	<= '0';	-- 0 ns	
		wait for 62 ns;	          
		input_spike	<= '1'; -- 62 ns
		wait for 96 ns;			          
		input_spike	<= '0'; -- 158 ns
		wait for 24 ns;			          
		input_spike	<= '1'; -- 182 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 194 ns
		wait for 12 ns;			          
		input_spike	<= '1'; -- 206 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 218 ns
		wait for 12 ns;			          
		input_spike	<= '1'; -- 230 ns
		wait for 12 ns;			          
		input_spike	<= '0'; -- 242 ns
		wait;
	end process input_spike_gen;




	-- stop
	stop_gen : process
	begin
		stop	<= '0';		-- 0 ns
		wait for 266 ns;
		stop	<= '1';		-- 266 ns
		wait for 12 ns;
		stop <= '0';		-- 278 ns
		wait;
	end process stop_gen;






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
	state_evaluation	: process(present_state, start, stop, exc_or,
					inh_or, exc_stop, inh_stop, input_spike,
					exceed_v_th)
	begin


		-- default case
		-- next_state	<= reset;

		case present_state is
			
			-- reset
			when reset =>
				next_state <= idle;

			-- idle
			when idle =>

				if start = '1'
				then
					next_state <= exp_decay;
				else
					next_state <= idle;
				end if;
			

			-- exp_decay
			when exp_decay =>

				if stop = '1'
				then
					next_state <= idle;

				elsif exc_or = '1'
				then
					next_state <= no_exc_spike;

				elsif inh_or = '1'
				then
					next_state <= no_inh_spike;

				else
					next_state <= exp_decay;

				end if;

			-- no_exc_spike
			when no_exc_spike =>

				if exc_stop = '1'
				then
					if inh_or = '1'
					then
						next_state <= no_inh_spike;

					elsif exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= exc_spike;

				else
					next_state <= no_exc_spike;
				end if;
				

			-- exc_spike
			when exc_spike =>

				if exc_stop = '1'
				then
					if inh_or = '1'
					then
						next_state <= no_inh_spike;

					elsif exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= exc_spike;

				else
					next_state <= no_exc_spike;
				end if;


			-- no_inh_spike
			when no_inh_spike =>
				
				if inh_stop = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= inh_spike;

				else
					next_state <= no_inh_spike;
				end if;



			-- inh_spike
			when inh_spike =>

				if inh_stop = '1'
				then
					if exceed_v_th = '1'
					then
						next_state <= fire;

					else
						next_state <= exp_decay;
					end if;

				elsif input_spike = '1'
				then
					next_state <= inh_spike;

				else
					next_state <= no_inh_spike;
				end if;

				

			

			-- fire
			when fire =>

				if stop = '1'
				then
					next_state <= idle;

				elsif exc_or = '1'
				then
					next_state <= no_exc_spike;

				else
					next_state <= no_inh_spike;

				end if;

			-- default case
			when others =>
				next_state <= reset;


		end case;

	end process state_evaluation;



	output_evaluation	: process(present_state)
	begin

		-- default values
		update_sel	<= "00";
		add_or_sub	<= '0';
		v_update	<= '1';
		v_en		<= '1';
		v_rst_n		<= '1';
		out_spike	<= '0';
		neuron_ready	<= '0';

		case present_state is

			-- reset
			when reset =>
				v_en 		<= '0';
				v_rst_n		<= '0';

			-- idle
			when idle =>
				neuron_ready	<= '1';
				v_en 		<= '0';
				v_rst_n		<= '0';

			-- exp_decay
			when exp_decay =>
				add_or_sub	<= '1';
				update_sel	<= "01";
				

			-- no_exc_spike
			when no_exc_spike	=>
				v_en		<= '0';

			-- exc_spike
			when exc_spike =>
				update_sel	<= "10";
				v_en		<= '1';

				
			-- no_inh_spike
			when no_inh_spike	=>
				v_en		<= '0';


			-- inh_spike
			when inh_spike		=>
				update_sel	<= "11";
				v_en		<= '1';


			-- fire
			when fire =>
				v_update	<= '0';
				out_spike	<= '1';

			-- default case
			when others =>
				v_en 		<= '0';
				v_rst_n		<= '0';

		end case;

	end process output_evaluation;
	




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
			sel				=> update_sel,
			in0				=> (others => '0'),
			in1				=> v_shifted,
			in2(N_weight-1 downto 0)	=> exc_weight,
			in2(N-1 downto N_weight)	=> (others => '0'),
			in3				=> inh_weight,
		                               
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

			-- output
			cmp_out	=> exceed_v_th
		);




end architecture behaviour;
