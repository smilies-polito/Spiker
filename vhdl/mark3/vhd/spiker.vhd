library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spiker is

	port(
					
		-- input
		clk			: in std_logic;
		rst_n			: in std_logic;	
		cnt_rst_n		: in std_logic;
		start			: in std_logic;	
		lfsr_en			: in std_logic;
		load_n			: in std_logic;
		reg_addr		: in std_logic_vector(9 downto 0);
		pixel_in		: in std_logic_vector(7 downto 0);
		lfsr_in			: in std_logic_vector(12 downto 0);
		outSel			: in std_logic_vector(8 downto 0);
		weights_in		: in std_logic_vector(17 downto 0);
		wraddr			: in std_logic_vector(15 downto 0);
		bram_sel		: in std_logic_vector(5 downto 0);

		-- input parameters
		v_th_value		: in signed(15 downto 0);		

		-- output
		sample			: out std_logic;
		ready			: out std_logic;
		spikesCount		: out signed(8 downto 0)
	);

end entity spiker;



architecture behaviour of spiker is

	type out_matrix is array(399 downto 0) of std_logic_vector(8 downto 0);

	constant neuronParallelism	: integer := 16;
	constant weightParallelism	: integer := 5;
	constant N_inputs		: integer := 784;
	constant N_exc_cnt		: integer := 11;
	constant N_neurons		: integer := 400;
	constant N_inh_cnt		: integer := 10;
	constant N_cycles		: integer := 3500;
	constant N_cycles_cnt		: integer := 12;
	constant shift			: integer := 1;

	constant v_reset		: integer := 5*2**10;
	constant inh_weight		: integer := -15;

	signal input_spikes		: std_logic_vector(N_inputs-1 downto 0);
	signal exc_weights		: std_logic_vector(N_neurons*weightParallelism-1 downto 0);
	signal exc_cnt			: std_logic_vector(N_exc_cnt-1 
						downto 0);
	signal outSpikes		: std_logic_vector(N_neurons-1 
						downto 0);



	signal v_reset_sig		: signed(neuronParallelism-1 downto 0);	
	signal inh_weight_sig		: signed(neuronParallelism-1 downto 0);	
	signal N_inputs_sig		: std_logic_vector(N_exc_cnt-1 downto 0);	
	signal N_neurons_sig		: std_logic_vector(N_inh_cnt-1 downto 0);	
	signal N_cycles_sig		: std_logic_vector(N_cycles_cnt-1 downto 0);	


	signal cnt_out			: out_matrix;
	signal outCounts		: std_logic_vector(9*2**9-1 downto 0);


	signal rdaddr			: std_logic_vector(15 downto 0);
	signal rden			: std_logic;
	signal weights_input		: std_logic_vector(71 downto 0);

	component input_layer is

		port(
			-- input
			clk		: in std_logic;
			en		: in std_logic;
			load_n		: in std_logic;
			reg_addr	: in std_logic_vector(9 downto 0);
			pixel_in	: in std_logic_vector(7 downto 0);
			lfsr_in		: in std_logic_vector(12 downto 0);

			-- output
			spikes		: out std_logic_vector(783 downto 0)
		);

	end component input_layer;




	component layer is

		generic(

			-- internal parallelism
			parallelism		: integer := 16;
			weightParallelism	: integer := 5;

			-- excitatory spikes
			input_parallelism	: integer := 784;
			N_exc_cnt		: integer := 11;

			-- inhibitory spikes
			layer_size		: integer := 400;
			N_inh_cnt		: integer := 10;

			-- elaboration steps
			N_cycles_cnt		: integer := 12;

			-- exponential decay shift
			shift			: integer := 1
				
		);

		port(
			-- input
			clk			: in std_logic;
			rst_n			: in std_logic;	
			start			: in std_logic;	
			input_spikes		: in std_logic_vector
							(input_parallelism-1 downto 0);

			-- input parameters
			v_th_value		: in signed(parallelism-1 downto 0);		
			v_reset			: in signed(parallelism-1 downto 0);	
			inh_weight		: in signed(parallelism-1 downto 0);		
			exc_weights		: in signed
						(layer_size*weightParallelism-1 downto 0);

			-- number of inputs, neurons and cycles
			N_inputs		: in std_logic_vector
							(N_exc_cnt-1 downto 0);
			N_neurons		: in std_logic_vector
							(N_inh_cnt-1 downto 0);
			N_cycles		: in std_logic_vector
							(N_cycles_cnt-1 downto 0);

			-- output
			exc_cnt			: out std_logic_vector
							(N_exc_cnt-1 downto 0);
			out_spikes		: out std_logic_vector
							(layer_size-1 downto 0);
			sample			: out std_logic;
			ready			: out std_logic
		);

	end component layer;


	component weights_bram is

		port(
			-- input
			clk		: in std_logic;
			di		: in std_logic_vector(71 downto 0);
			rdaddr		: in std_logic_vector(15 downto 0);
			rden		: in std_logic;
			wraddr		: in std_logic_vector(15 downto 0);
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



	component generic_mux_signed is

		generic(

			-- parallelism of the inputs selector
			N_sel	: integer := 4;

			-- parallelism
			N		: integer := 16
				
		);

		port(
			-- input
			input_data	: in signed(N*(2**N_sel)-1 downto 0);
			sel		: in std_logic_vector(N_sel-1 downto 0);

			-- output
			output_data	: out signed(N-1 downto 0)
		);

	end component generic_mux_signed;



begin

	v_reset_sig	<= to_signed(v_reset, neuronParallelism);
	inh_weight_sig	<= to_signed(inh_weight, neuronParallelism);


	N_inputs_sig	<= std_logic_vector(to_signed(N_inputs,
				N_exc_cnt));
	N_neurons_sig	<= std_logic_vector(to_signed(N_neurons,
				N_inh_cnt));
	N_cycles_sig	<= std_logic_vector(to_signed(N_cycles,
				N_cycles_cnt));

	process(weights_in)
	begin

		for i in 0 to 3
		loop

			weights_input((i+1)*18-1 downto i*18)	<= weights_in;

		end loop;

	end process;


	poissonLayer	: input_layer 

		port map(
			-- input
			clk		=> clk,
			en		=> lfsr_en,
			load_n		=> load_n,
			reg_addr	=> reg_addr,
			pixel_in	=> pixel_in,
			lfsr_in		=> lfsr_in,

			-- output
			spikes		=> input_spikes
		);




	layer_1		: layer 

		generic map(

			-- internal parallelism
			parallelism		=> neuronParallelism,
			weightParallelism	=> weightParallelism,

			-- excitatory spikes
			input_parallelism	=> N_inputs,
			N_exc_cnt		=> N_exc_cnt,

			-- inhibitory spikes
			layer_size		=> N_neurons,
			N_inh_cnt		=> N_inh_cnt,

			-- elaboration steps
			N_cycles_cnt		=> N_cycles_cnt,

			-- exponential decay shift
			shift			=> shift
				
		)

		port map(
			-- input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
			input_spikes		=> input_spikes,
						

			-- input parameters
			v_th_value		=> v_th_value,
			v_reset			=> v_reset_sig,
			inh_weight		=> inh_weight_sig,
			exc_weights		=> signed(exc_weights),
						

			-- number of inputs, neurons and cycles
			N_inputs		=> N_inputs_sig,
			N_neurons		=> N_neurons_sig,
			N_cycles		=> N_cycles_sig,
						

			-- output
			exc_cnt			=> exc_cnt,
						
			out_spikes		=> outSpikes,
						
			sample			=> sample,
			ready			=> ready
		);



	output_layer	: for i in 0 to 399
		generate

			out_counter	: cnt
				generic map(
					N		=> 9
				)

				port map(
					-- input
					clk		=> clk,
					cnt_en		=> outSpikes(i),
					cnt_rst_n	=> cnt_rst_n,

					-- output
					cnt_out		=> cnt_out(i)
				);

		end generate output_layer;


	megaOutSignal	: process(cnt_out)
		begin

			outCounts(9*2**9-1 downto 400*9)	<= (others =>
					'0');

			for i in 0 to 399
			loop

				outCounts((i+1)*9-1 downto i*9) 	<=
					cnt_out(i);

			end loop;

		end process megaOutSignal;



	output_mux	: generic_mux_signed

		generic map(

			-- parallelism of the inputs selector
			N_sel		=> 9,

			-- parallelism
			N		=> 9				
		)

		port map(
			-- input
			input_data	=> signed(outCounts),		
			sel		=> outSel,

			-- output
			output_data	=> spikesCount
		);

	weights_memory	: weights_bram

		port map(
			-- input
			clk		=> clk,
			di		=> weights_input,
			rdaddr		=> rdaddr,
			rden		=> rden,
			wraddr		=> wraddr,
			bram_sel	=> bram_sel,

			-- output
			do		=> exc_weights
					
		);





	end architecture behaviour;
