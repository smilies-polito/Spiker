library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spiker is

	generic(
		-- int parallelism
		parallelism		: integer := 16;
		weights_bit_width	: integer := 5;

		-- memory parameters
		word_length		: integer := 36;
		rdwr_addr_length	: integer := 10;
		we_length		: integer := 4;
		N_bram			: integer := 58;
		bram_addr_length	: integer := 6;

		-- input spikes
		N_inputs		: integer := 784;

		-- must be one bit larger that the parallelism required to count
		-- up to N_inputs
		N_inputs_cnt		: integer := 11;

		-- inhibitory spikes
		N_neurons		: integer := 400;

		-- must be one bit larger that the parallelism required to count
		-- up to N_neurons
		N_neurons_cnt		: integer := 10;

		-- Cycles counter
		N_cycles_cnt		: integer := 12;

		-- exponential decay shift
		shift			: integer := 10;

		-- Input interface bit width
		input_data_bit_width	: integer := 8;
		lfsr_bit_width		: integer := 16;

		-- Output counters parallelism
		N_out			: integer := 16
	);

	port(

		-- Common signals ---------------------------------------------
		clk			: in std_logic;
		rst_n			: in std_logic;	

		-- Input interface signals ------------------------------------
		init_lfsr		: in std_logic;
		seed			: in unsigned(lfsr_bit_width-1 downto 0);

		-- Layer signals ----------------------------------------------  

		-- control input
		start			: in std_logic;	
		init_v_th		: in std_logic;

		-- address to select the neurons
		v_th_addr		: in std_logic_vector(N_neurons_cnt-1
						  downto 0);

		-- data input
		input_data		: in unsigned(N_inputs*
						input_data_bit_width-1 
						downto 0);

		-- input parameters
		v_th_value		: in signed(parallelism-1 downto 0);		
		v_reset			: in signed(parallelism-1 downto 0);	
		inh_weight		: in signed(parallelism-1 downto 0);		

		-- terminal counters 
		N_inputs_tc		: in std_logic_vector
						(N_inputs_cnt-1 downto 0);
		N_neurons_tc		: in std_logic_vector
						(N_neurons_cnt-1 downto 0);
		N_cycles_tc		: in std_logic_vector(N_cycles_cnt-1
						downto 0);

		-- output
		ready			: out std_logic;
		cnt_out			: out std_logic_vector(N_neurons*N_out-1
						downto 0);


		-- Memory signals --------------------------------------------- 
		-- input
		di		: in std_logic_vector(35 downto 0);
		rden		: in std_logic;
		wren		: in std_logic;
		wraddr		: in std_logic_vector(9 downto 0);
		bram_sel	: in std_logic_vector(5 downto 0)
	);

end entity spiker;



architecture behaviour of spiker is


	-- Memory signals: input
	signal rdaddr			: std_logic_vector(10 downto 0);

	-- Memory signals: output
	signal exc_weights		:
		std_logic_vector(N_neurons*weights_bit_width-1 downto 0);

	-- Layer signals: input
	signal stop			: std_logic;	
	signal input_spikes		: std_logic_vector
						(N_inputs-1 downto 0);

	-- Layer signals: output
	signal out_spikes		: std_logic_vector(N_neurons-1 downto 0);
	signal sample			: std_logic;

	-- Output counters signals
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;	
	signal cycles_cnt		: std_logic_vector(N_cycles_cnt-1 downto
						0);
	signal cnt_out_rst_n		: std_logic;
	


	component weights_bram is

		generic(
			word_length		: integer := 36;
			rdwr_addr_length	: integer := 10;
			we_length		: integer := 4;
			N_neurons		: integer := 400;
			weights_bit_width	: integer := 5;
			N_bram			: integer := 58;
			bram_addr_length	: integer := 6
		);

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(word_length-1 downto 0);
			rst_n		: in std_logic;
			rdaddr		: in std_logic_vector(rdwr_addr_length-1 
						downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(rdwr_addr_length-1
						downto 0);
			bram_sel	: in std_logic_vector(weights_bit_width-1 
						downto 0);

			-- output
			do		: out std_logic_vector(N_neurons*
						weights_bit_width-1 downto 0)
					
		);

	end component weights_bram;


	component layer is

		generic(

			-- int parallelism
			parallelism		: integer := 16;
			weights_bit_width	: integer := 5;

			-- input spikes
			N_inputs		: integer := 784;

			-- must be one bit larger that the parallelism required to count
			-- up to N_inputs
			N_inputs_cnt		: integer := 11;

			-- inhibitory spikes
			N_neurons		: integer := 400;

			-- must be one bit larger that the parallelism required to count
			-- up to N_neurons
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
			v_th_addr		: in std_logic_vector(N_neurons_cnt-1
							  downto 0);

			-- data input
			input_spikes		: in std_logic_vector
							(N_inputs-1 downto 0);

			-- input parameters
			v_th_value		: in signed(parallelism-1 downto 0);		
			v_reset			: in signed(parallelism-1 downto 0);	
			inh_weight		: in signed(parallelism-1 downto 0);		
			exc_weights		: in signed
							(N_neurons*weights_bit_width-1
							 downto 0);

			-- terminal counters 
			N_inputs_tc		: in std_logic_vector
							(N_inputs_cnt-1 downto 0);
			N_neurons_tc		: in std_logic_vector
							(N_neurons_cnt-1 downto 0);

			-- output
			out_spikes		: out std_logic_vector
							(N_neurons-1 downto 0);
			ready			: out std_logic;
			sample			: out std_logic;
			cycles_cnt_rst_n	: out std_logic;	
			cycles_cnt_en		: out std_logic;	

			-- output address to select the excitatory weights
			exc_cnt			: out std_logic_vector
							(N_inputs_cnt-1 downto 0)
		);

	end component layer;


	component input_interface is

		generic(
			bit_width	: integer := 16;
			N_inputs	: integer := 784
		);

		port(
			-- control input
			clk		: in std_logic;
			load		: in std_logic;
			update		: in std_logic;

			-- data input
			seed		: in unsigned(bit_width-1 downto 0);
			input_data	: in unsigned(N_inputs*bit_width-1 downto 0);

			-- output
			output_spikes	: out std_logic_vector(N_inputs-1 downto 0)
		);

	end component input_interface;


	component out_interface is

		generic(
			bit_width	: integer := 16;	
			N_neurons	: integer := 400
		);

		port(
			-- control input
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;

			-- data input
			spikes		: in std_logic_vector(N_neurons-1 downto 0);

			-- output
			cnt_out		: out std_logic_vector(N_neurons*bit_width-1
						downto 0)

		);

	end component out_interface;


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

	cnt_out_rst_n	<= not start;


	weights_memory	: weights_bram 

		generic map(
			word_length		=> word_length,
			rdwr_addr_length	=> rdwr_addr_length,
			we_length		=> we_length,
			N_neurons		=> N_neurons,
			weights_bit_width	=> weights_bit_width,
			N_bram			=> N_bram,
			bram_addr_length	=> bram_addr_length
		)

		port map(
			-- input
			clk		=> clk,
			di		=> di,
			rst_n		=> rst_n,
			rdaddr		=> rdaddr(9 downto 0),
			rden		=> rden,
			wren		=> wren,
			wraddr		=> wraddr,
			bram_sel	=> bram_sel,

			-- output
			do		=> exc_weights
					
		);


	complete_layer	: layer
		generic map(

			-- int parallelism
			parallelism		=> parallelism,
			weights_bit_width	=> weights_bit_width,

			-- input spikes
			N_inputs		=> N_inputs,

			-- must be one bit larger that the parallelism required to count
			-- up to N_inputs
			N_inputs_cnt		=> N_inputs_cnt,

			-- inhibitory spikes
			N_neurons		=> N_neurons,

			-- must be one bit larger that the parallelism required to count
			-- up to N_neurons
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
			exc_weights		=> signed(exc_weights),
					           		
					           		
                                                                   
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
			exc_cnt			=> rdaddr
		);


	input_layer	: input_interface

		generic map(
			bit_width	=> N_out,
			N_inputs	=> N_inputs
		)

		port map(
			-- control input
			clk		=> clk,
			load		=> init_lfsr,
			update		=> sample,

			-- data input
			seed		=> seed,
			input_data	=> input_data,

			-- output
			output_spikes	=> input_spikes
		);

	output_counters	: out_interface 

		generic map(
			bit_width	=> N_out,
			N_neurons	=> N_neurons
		)

		port map(
			-- control input
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,

			-- data input
			spikes		=> out_spikes,

			-- output
			cnt_out		=> cnt_out
					

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

end architecture behaviour;
