library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity debug_layer_tb is
end entity debug_layer_tb;

architecture test of debug_layer_tb is

	-- Parallelisms
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

	-- Number of elaboration cycles
	constant N_cycles		: integer := 3500;
	
	-- Cycles cnt
	constant N_cycles_cnt		: integer := 12;
	

	-- Constant parameters
	constant v_reset_int		: integer := 5*2**3; 	  
	constant inh_weight_int	 	: integer := -15*2**3; 


	--File names
	constant weights_filename	: string	:= "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/Miei/spiker/vhdl/mark3/sim/"&
		"hyperparameters/weights.mem";

	constant thresholds_filename	: string	:= "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/Miei/spiker/vhdl/mark3/sim/"&
		"hyperparameters/thresholds.init";

	constant inputs_filename	: string	:= "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/Miei/spiker/vhdl/mark3/"&
		"sim/inputOutput/inputSpikes.txt";

	constant out_spikes_filename	: string	:= "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/Miei/spiker/vhdl/mark3/"&
		"sim/inputOutput/vhdlOutSpikes.txt";

	constant membrane_filename	: string	:= "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/Miei/spiker/vhdl/mark3/"&
		"sim/inputOutput/vhdlMembrane.txt";

	
	constant out_weights_filename	: string	:= "/home/alessio/"&
		"OneDrive/Dottorato/Progetti/SNN/Miei/spiker/vhdl/mark3/"&
		"sim/inputOutput/outWeights.txt";

	-- BRAM parameters
	constant weightsWord		: integer := 36;
	constant bram_addr_length	: integer := 6;
	constant weights_addr_length	: integer := 10;
	constant N_bram			: integer := 58;






	-- Io files control signals
	signal weights_rden		: std_logic;
	signal thresholds_rden		: std_logic;
	signal write_out		: std_logic;
	signal dummy_addr		: std_logic_vector(0 downto 0);




	-- Common signals
	signal clk			: std_logic;
	signal rst_n			: std_logic;




	-- Layer signals: control input
	signal start			: std_logic;	
	signal stop			: std_logic;	
	signal init_v_th		: std_logic;

	
	-- Layer signals: address to select the neurons
	signal v_th_addr		: std_logic_vector(N_neurons_cnt-1
						downto 0);

	-- Layer signals: data input
	signal input_spikes		: std_logic_vector
					     (N_inputs-1 downto 0);

	-- Layer signals: input parameters
	signal v_th_value		: signed(parallelism-1 downto 0);	
	signal v_reset			: signed(parallelism-1 downto 0);	
	signal inh_weight		: signed(parallelism-1 downto 0);	
	signal exc_weights		: signed(N_neurons*weightsParallelism-1
						downto 0);

	-- Terminal counters
	signal N_inputs_tc		: std_logic_vector
						(N_inputs_cnt-1 downto 0);
	signal N_neurons_tc		: std_logic_vector
						(N_neurons_cnt-1 downto 0);
	signal N_cycles_tc		: std_logic_vector
						(N_cycles_cnt-1 downto 0);

	-- Layer signals: output
	signal out_spikes		: std_logic_vector(N_neurons-1 downto
						0);
	signal ready			: std_logic;
	signal sample			: std_logic;
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;	


	-- Layer signals: output address to select the excitatory weights
	signal exc_cnt			: std_logic_vector(N_inputs_cnt-1 
						downto 0);

	-- Layer signals: debug output
	signal v_out			: signed(parallelism-1 downto 0);





	-- Memory signals: input
	signal input_weights		: std_logic_vector(35 downto 0);
	signal rden			: std_logic;
	signal wren			: std_logic;
	signal wraddr			: std_logic_vector(weights_addr_length-1
						downto 0);
	signal bram_sel			: std_logic_vector(bram_addr_length-1
						downto 0);
						

	-- Cycles counter signals
	signal cycles_cnt		: std_logic_vector(N_cycles_cnt-1 
						downto 0);




		
	component load_file is

		generic(
			word_length		: integer := 36;
			bram_addr_length	: integer := 6;
			addr_length		: integer := 16;
			N_bram			: integer := 58;
			N_words			: integer := 784;
			weights_filename	: string := "/home/alessio/"&
			"OneDrive/Dottorato/Progetti/SNN/spiker/vhdl/mark3/"&
			"hyperparameters/weights.mem"
		);

		port(
			-- input
			clk			: in std_logic;
			rden			: in std_logic;

			-- output
			di			: out std_logic_vector(
							word_length-1 downto 0);
			bram_addr		: out std_logic_vector(
							bram_addr_length - 1 
							downto 0);
			wraddr			: out std_logic_vector(
							addr_length-1 downto 0);
			wren			: out std_logic
		);

	end component load_file;



	component debug_layer is

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

			-- exponential decay shift
			shift			: integer := 10
		);

		port(
			-- control input
			clk			: in std_logic;
			rst_n			: in std_logic;	
			start			: in std_logic;	
			stop			: in std_logic;	
			init_v_th		: in std_logic;

			-- address to select the neurons
			v_th_addr		: in std_logic_vector(
							N_neurons_cnt-1 downto
							0);

			-- data input
			input_spikes		: in std_logic_vector
							(N_inputs-1 downto 0);

			-- input parameters
			v_th_value		: in signed(parallelism-1 downto
							0);		
			v_reset			: in signed(parallelism-1 downto
							0);	
			inh_weight		: in signed(parallelism-1 downto
							0);		
			exc_weights		: in signed(N_neurons*
							weightsParallelism-1
						       	downto 0);

			-- terminal counters 
			N_inputs_tc		: in std_logic_vector
							(N_inputs_cnt-1 
							downto 0);
			N_neurons_tc		: in std_logic_vector
							(N_neurons_cnt-1 
							downto 0);

			-- output
			out_spikes		: out std_logic_vector
							(N_neurons-1 downto 0);
			ready			: out std_logic;
			sample			: out std_logic;
			cycles_cnt_rst_n	: out std_logic;	
			cycles_cnt_en		: out std_logic;	

			-- output address to select the excitatory weights
			exc_cnt			: out std_logic_vector
							(N_inputs_cnt-1 
							downto 0);

			-- debug output
			v_out			: out signed(parallelism-1
							downto 0)
		);

	end component debug_layer;


	component weights_bram is

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(35 downto 0);
			rst_n		: in std_logic;
			rdaddr		: in std_logic_vector(9 downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(9 downto 0);
			bram_sel	: in std_logic_vector(5 downto 0);

			-- output
			do		: out std_logic_vector(400*5-1 downto 0)
					
		);

	end component weights_bram;


	component cnt is

		generic(
			N		: integer := 8		
		);

		port(
			-- input
			clk		: in std_logic;
			cnt_en		: in std_logic;
			cnt_rst_n	: in std_logic;

			-- output
			cnt_out		: out std_logic_vector(N-1 downto 0)		
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


begin

	v_reset			<= to_signed(v_reset_int, v_reset'length);
	inh_weight		<= to_signed(inh_weight_int, inh_weight'length);

	N_inputs_tc		<= std_logic_vector(to_signed(N_inputs,
			       		N_inputs_tc'length));
	N_neurons_tc		<= std_logic_vector(to_signed(N_neurons,
					N_neurons_tc'length));
	N_cycles_tc		<= std_logic_vector(to_signed(N_cycles,
					N_cycles_tc'length));

	v_th_addr(N_neurons_cnt-1) <= '0';
	dummy_addr	<= "0";



	-- initialize weights
	init_weights	: load_file 

		generic map(
			word_length		=> weightsWord,
			bram_addr_length	=> bram_addr_length,
			addr_length		=> weights_addr_length,
			N_bram			=> N_bram,
			N_words			=> N_inputs,
			weights_filename	=> weights_filename
		)

		port map(
			-- input
			clk			=> clk,
			rden			=> weights_rden,

			-- output
			di			=> input_weights,
			bram_addr		=> bram_sel,
			wraddr			=> wraddr,
			wren			=> wren
		);


	-- initialize thresholds
	init_thresholds : load_file 

		generic map(
			word_length		=> parallelism,
			bram_addr_length	=> 1,
			addr_length		=> N_neurons_cnt-1,
			N_bram			=> 1,
			N_words			=> N_neurons,
			weights_filename	=> thresholds_filename
		)

		port map(
			-- input
			clk			=> clk,
			rden			=> thresholds_rden,

			-- output
			std_logic_vector(di)	=> v_th_value,
			bram_addr		=> dummy_addr,
			wraddr			=> v_th_addr(N_neurons_cnt-2
							downto 0),
			wren			=> init_v_th 
		);



	-- read inputs from file
	read_inputs	: process(clk, sample)

		file inputs_file	: text open read_mode is
			inputs_filename;

		variable read_line	: line;
		variable inputs_var	: std_logic_vector(N_inputs-1 
						downto 0);

	begin

		if clk'event and clk = '1'
		then
			if sample = '1'
			then
				if not endfile(inputs_file)
				then

					-- Read line from file
					readline(inputs_file, read_line);
					read(read_line, inputs_var);

					-- Associate line to data input
					input_spikes	<= inputs_var;

				end if;
			end if;
		end if;	
	end process read_inputs;



	-- Store output spikes on file
	store_spikes	: process(clk, sample)

		file output_file	: text open write_mode is
			out_spikes_filename;

		variable write_line	: line;

	begin

		if clk'event and clk = '1'
		then
			if sample = '1'
			then

				write(write_line, out_spikes(0));
				writeline(output_file, write_line);

			end if;
		end if;	

	end process store_spikes;



	-- Store membrane voltage of the first neuron on file
	store_membrane	: process(clk, sample)

		file output_file	: text open write_mode is
			membrane_filename;

		variable write_line	: line;

	begin

		if clk'event and clk = '1'
		then
			if sample = '1'
			then

				write(write_line, to_integer(v_out));
				writeline(output_file, write_line);

			end if;
		end if;	

	end process store_membrane;


	-- Store weights evolution
	store_weights_evolution	: process(clk, sample)

		file output_file	: text open write_mode is
			out_weights_filename;

		variable write_line	: line;

	begin

		if clk'event and clk = '1'
		then

			write(write_line, sample);
			write(write_line, string'(" "));
			write(write_line, exc_cnt);
			write(write_line, string'(" "));
			write(write_line, std_logic_vector(exc_weights));
			
			writeline(output_file, write_line);

		end if;	

	end process store_weights_evolution;




	-- weights read enable
	weights_rden_gen	: process
	begin
		weights_rden <= '0';
		wait for 100 ns;
		weights_rden <= '1';
		wait for 1 ms;
		weights_rden <= '0';
		wait;
	end process weights_rden_gen;

	-- thresholds read enable
	thresholds_rden_gen	: process
	begin
		thresholds_rden <= '0';
		wait for 1.1 ms;
		thresholds_rden <= '1';
		wait for 10 us;
		thresholds_rden <= '0';
		wait;
	end process thresholds_rden_gen;

	-- enable output write on file
	write_out_gen	: process
	begin
		write_out <= '0';
		wait for 25 ms;
		write_out <= '1';
		wait for 20 ns;
		write_out <= '0';
		wait;
	end process write_out_gen;





	-- clock
	clk_gen		: process
	begin
		clk <= '0';
		wait for 10 ns;
		clk <= '1';
		wait for 10 ns;
	end process clk_gen;

	-- reset (active low)
	rst_n_gen	: process
	begin
		rst_n <= '1';
		wait for 42 ns;
		rst_n <= '0';
		wait for 10 ns;
		rst_n <= '1';
		wait;
	end process rst_n_gen;

	-- read enable
	rden_gen	: process
	begin
		rden <= '0';
		wait for 1.2 ms;
		rden <= '1';
		wait;
	end process rden_gen;

	-- start generation
	start_gen	: process
	begin
		start <= '0';
		wait for 1.3 ms;
		start <= '1';
		wait for 20 ns;
		start <= '0';
		wait;
	end process start_gen;


	
	layer	: debug_layer 
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

			-- exponential decay shift
			shift			=> shift
		)

		port map(
			-- control input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			stop			=> stop,
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
                                                                   
			-- output
			out_spikes		=> out_spikes,
			ready			=> ready,
			sample			=> sample,
			cycles_cnt_rst_n	=> cycles_cnt_rst_n,
			cycles_cnt_en		=> cycles_cnt_en,

			-- output address to select the excitatory weights
			exc_cnt			=> exc_cnt,

			-- debug output
			v_out			=> v_out
		);


	synapses_weights	: weights_bram 
		port map(
			-- input
			clk		=> clk,
			di		=> input_weights,
			rst_n		=> rst_n,
			rdaddr		=> exc_cnt(9 downto 0),
			rden		=> rden,
			wren		=> wren,
			wraddr		=> wraddr,
			bram_sel	=> bram_sel,

			-- output
			signed(do)		=> exc_weights
					
		);


	cycles_counter	: cnt
		generic map(
			N		=> N_cycles_cnt
		)

		port map(
			-- input
			clk		=> clk,
			cnt_en		=> cycles_cnt_en,
			cnt_rst_n	=> cycles_cnt_rst_n,
							   
			-- output
			cnt_out		=> cycles_cnt
		);


	cycles_stop	: cmp_eq 
		generic map(
			N	=> N_cycles_cnt	
		)

		port map(
			-- input
			in0	=> cycles_cnt,
			in1	=> N_cycles_tc,

			-- output
			cmp_out	=> stop
		);


end architecture test;
