library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spiker_tb is
end entity spiker_tb;

architecture test of spiker_tb is

	-- Int parallelism
	constant parallelism		: integer := 16;
	constant weightsParallelism	: integer := 5;

	-- Input spikes
	constant N_inputs		: integer := 784;

	-- Must be one bit larger that the parallelism required to count
	-- up to N_inputs
	constant N_inputs_cnt		: integer := 11;

	-- Inhibitory spikes
	constant N_neurons		: integer := 400;

	-- Must be one bit larger that the parallelism required to count
	-- up to N_neurons
	constant N_neurons_cnt		: integer := 10;

	-- Exponential decay shift
	constant shift			: integer := 10;

	-- Output counters parallelism
	constant N_out			: integer := 16;
	
	-- Cycles cnt
	constant N_cycles_cnt		: integer := 12;

	-- Common signals
	signal clk			: std_logic;
	signal rst_n			: std_logic;
	
	-- Memory signals: input
	signal di			: std_logic_vector(35 downto 0);
	signal rden			: std_logic;
	signal wren			: std_logic;
	signal wraddr			: std_logic_vector(9 downto 0);
	signal bram_sel			: std_logic_vector(5 downto 0);

	-- Layer signals: input
	signal start			: std_logic;	
	signal init_v_th		: std_logic;
	signal v_th_addr		: std_logic_vector(N_neurons_cnt-1
						downto 0);
	signal input_spikes		: std_logic_vector
					     (N_inputs-1 downto 0);
	signal v_th_value		: signed(parallelism-1 downto 0);		
	signal v_reset			: signed(parallelism-1 downto 0);	
	signal inh_weight		: signed(parallelism-1 downto 0);		
	signal exc_weights		: signed
						(N_neurons*weightsParallelism-1
						 downto 0);
	signal N_inputs_tc		: std_logic_vector
						(N_inputs_cnt-1 downto 0);
	signal N_neurons_tc		: std_logic_vector
						(N_neurons_cnt-1 downto 0);
	signal N_cycles_tc		: std_logic_vector
						(N_cycles_cnt-1 downto 0);

	-- Layer signals: output
	signal ready			: std_logic;
	signal sample			: std_logic;


	-- Output counters signal
	signal cnt_out			: std_logic_vector(N_neurons*N_out-1
						downto 0);

	


	component spiker is

		generic(
			-- int parallelism
			parallelism		: integer := 16;
			weightsParallelism	: integer := 5;

			-- input spikes
			N_inputs		: integer := 784;

			-- must be one bit larger that the parallelism required
			-- to count up to N_inputs
			N_inputs_cnt		: integer := 11;

			-- inhibitory spikes
			N_neurons		: integer := 400;

			-- must be one bit larger that the parallelism required
			-- to count up to N_neurons
			N_neurons_cnt		: integer := 10;

			-- Cycles counter
			N_cycles_cnt		: integer := 12;

			-- exponential decay shift
			shift			: integer := 10;

			-- Output counters parallelism
			N_out			: integer := 16
		);

		port(

			-- Common signals --------------------------------------
			clk			: in std_logic;
			rst_n			: in std_logic;	

			-- Layer signals ---------------------------------------  

			-- control input
			start			: in std_logic;	
			init_v_th		: in std_logic;

			-- address to select the neurons
			v_th_addr		: in std_logic_vector
							(N_neurons_cnt-1
							  downto 0);

			-- data input
			input_spikes		: in std_logic_vector
							(N_inputs-1 downto 0);

			-- input parameters
			v_th_value		: in signed(parallelism-1 
							downto 0);		
			v_reset			: in signed(parallelism-1 
							downto 0);	
			inh_weight		: in signed(parallelism-1 
							downto 0);		
			exc_weights		: in signed(N_neurons
							*weightsParallelism-1
							 downto 0);

			-- terminal counters 
			N_inputs_tc		: in std_logic_vector
							(N_inputs_cnt-1 
							downto 0);
			N_neurons_tc		: in std_logic_vector
							(N_neurons_cnt-1 
							downto 0);
			N_cycles_tc		: in std_logic_vector(
							N_cycles_cnt-1
							downto 0);

			-- output
			ready			: out std_logic;
			sample			: out std_logic;


			-- Memory signals -------------------------------------- 
			-- input
			di		: in std_logic_vector(35 downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(9 downto 0);
			bram_sel	: in std_logic_vector(5 downto 0);

			-- Output counters signals
			cnt_out	: out std_logic_vector(N_neurons*N_out-1 
					downto 0)
		);

	end component spiker;

begin

	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns:
	end process clk_gen;

	rst_n_gen	: process
	begin
		rst_n <= '1';
		wait for 42 ns;
		rst_n <= '0';
		wait;
	end process rst_n_gen;


	-- load_weight	: process(clk)


	dut	: spiker
		generic map(
			-- int parallelism
			parallelism		=> parallelism,
			weightsParallelism	=> weightsParallelism,

			-- input spikes
			N_inputs		=> N_inputs,

			-- must be one bit larger that the parallelism required
			-- to count up to N_inputs
			N_inputs_cnt		=> N_inputs_cnt,

			-- inhibitory spikes
			N_neurons		=> N_neurons,

			-- must be one bit larger that the parallelism required
			-- to count up to N_neurons
			N_neurons_cnt		=> N_neurons_cnt,

			-- Cycles counter
			N_cycles_cnt		=> N_cycles_cnt,

			-- exponential decay shift
			shift			=> shift,

			-- Output counters parallelism
			N_out			=> N_out
		)

		port map(

			-- Common signals --------------------------------------
			clk			=> clk,
			rst_n			=> rst_n,

			-- Layer signals ---------------------------------------  

			-- control input
			start			=> start,
			init_v_th		=> init_v_th,

			-- address to select the neurons
			v_th_addr		=> v_th_addr,
                                                                   
			-- data input
			input_spikes		=> input_spikes,
                                                                   
			-- input parameters
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			exc_weights		=> exc_weights,
                                                                   
			-- terminal counters
			N_inputs_tc		=> N_inputs_tc,
			N_neurons_tc		=> N_neurons_tc,
			N_cycles_tc		=> N_cycles_tc,
                                                                   
			-- output
			ready			=> ready,
			sample			=> sample,


			-- Memory signals -------------------------------------- 
			-- input
			di		=> di,
			rden		=> rden,
			wren		=> wren,
			wraddr		=> wraddr,
			bram_sel	=> bram_sel,

			-- Output counters signals
			cnt_out		=> cnt_out	
		);

end architecture test;
