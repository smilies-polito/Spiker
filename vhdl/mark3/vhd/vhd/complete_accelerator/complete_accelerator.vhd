library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity complete_accelerator is

	generic(

		-- Interface bit-widths
		load_bit_width		: integer := 4;
		data_bit_width		: integer := 36;
		addr_bit_width		: integer := 10;
		sel_bit_width		: integer := 10;

		-- Internal bit-widths
		neuron_bit_width	: integer := 16;
		weights_bit_width	: integer := 15;
		bram_word_length	: integer := 36;
		bram_addr_length	: integer := 10;
		bram_sel_length		: integer := 1;
		bram_we_length		: integer := 4;
		input_data_bit_width	: integer := 8;
		lfsr_bit_width		: integer := 16;
		cnt_out_bit_width	: integer := 16;

		-- Network shape
		inputs_addr_bit_width	: integer := 2;
		neurons_addr_bit_width	: integer := 1;

		-- Must be 1 bit longer than what required to count to N_cycles
		cycles_cnt_bit_width	: integer := 6;

		-- Bram parameters
		N_bram			: integer := 1;
		N_weights_per_word	: integer := 2;

		-- Structure parameters
		N_inputs		: integer := 3;
		N_neurons		: integer := 2;

		-- Internal parameters
		shift			: integer := 10

	);

	port(

		-- Input
   		clk			: in std_logic;
		rst_n			: in std_logic;	
		start			: in std_logic;	
		addr			: in std_logic_vector(
						addr_bit_width-1
						downto 0);
		load			: in std_logic_vector(load_bit_width-1
						downto 0);
		data			: in std_logic_vector(
						data_bit_width-1 
						downto 0);
		sel			: in std_logic_vector(
						sel_bit_width-1
						downto 0);
		-- Output
		ready			: out std_logic;
		cnt_out			: out std_logic_vector(
						cnt_out_bit_width-1
						downto 0);

		-- Memory signals --------------------------------------------- 
		rden			: in std_logic
	);

end entity complete_accelerator;



architecture behaviour of complete_accelerator is



	-- Input data
	signal input_data_addr		: std_logic_vector(
						inputs_addr_bit_width-1
						downto 0);
					
	-- Internal thresholds
	signal v_th_addr		: std_logic_vector(
						neurons_addr_bit_width
						downto 0);

	-- Bram
	signal wraddr			: std_logic_vector(
						bram_addr_length-1
						downto 0);


	-- SIgnals to initialize the accelerator
	signal enable			: std_logic_vector(2**load_bit_width-1
						downto 0);

	signal init_lfsr		: std_logic;
	signal init_v_th		: std_logic;
	signal load_input_data		: std_logic;
	signal load_v_reset		: std_logic;
	signal load_inh_weight		: std_logic;
	signal wren			: std_logic;
	signal load_N_inputs_tc		: std_logic;
	signal load_N_neurons_tc	: std_logic;
	signal load_N_cycles_tc		: std_logic;


	-- Lfsr
	signal seed			: unsigned(lfsr_bit_width-1 
						downto 0);

	-- Input data
	signal input_data		: unsigned(input_data_bit_width-1 
						downto 0);

	-- Terminal counters: network shape. Must be 1 bit longer than what
	-- required to count to N_inputs and N_neurons
	signal N_inputs_tc_value	: std_logic_vector
						(inputs_addr_bit_width
						downto 0);
	signal N_neurons_tc_value	: std_logic_vector
						(neurons_addr_bit_width 
						downto 0);
	signal N_inputs_tc		: std_logic_vector
						(inputs_addr_bit_width 
						downto 0);
	signal N_neurons_tc		: std_logic_vector
						(neurons_addr_bit_width
						downto 0);

	-- Cycles terminal counter
	signal N_cycles_tc_value	: std_logic_vector(
						cycles_cnt_bit_width-1
						downto 0);
	signal N_cycles_tc		: std_logic_vector(
						cycles_cnt_bit_width-1
						downto 0);


	-- Threshold
	signal v_th_value		: signed(neuron_bit_width-1 downto 0);
	signal v_reset_value		: signed(neuron_bit_width-1 downto 0);	
	signal inh_weight_value		: signed(neuron_bit_width-1 downto 0);

	-- Input weights
	signal di			: std_logic_vector(bram_word_length-1 
						downto 0);

	-- Output counters selector
	signal cnt_out_sel		: std_logic_vector(
						neurons_addr_bit_width-1
						downto 0);

	signal bram_sel			: std_logic_vector(bram_sel_length-1
       						downto 0);


	signal buffered_data	: unsigned(N_inputs*input_data_bit_width-1 
					downto 0);
	signal padded_buff_data	: unsigned(N_inputs*lfsr_bit_width-1 
					downto 0);
	signal counters		: std_logic_vector(2**neurons_addr_bit_width*
					cnt_out_bit_width-1
					downto 0);
	signal v_reset		: signed(neuron_bit_width-1 downto 0);	
	signal inh_weight	: signed(neuron_bit_width-1 downto 0);		


	component spiker is

		generic(
			-- Bit-widths
			neuron_bit_width	: integer := 16;
			weights_bit_width	: integer := 5;
			bram_word_length	: integer := 36;
			bram_addr_length	: integer := 10;
			bram_sel_length		: integer := 6;
			bram_we_length		: integer := 4;
			input_data_bit_width	: integer := 8;
			lfsr_bit_width		: integer := 16;
			cnt_out_bit_width	: integer := 16;

			-- Network shape
			inputs_addr_bit_width	: integer := 10;
			neurons_addr_bit_width	: integer := 9;

			-- Must be 1 bit longer than what required to count to N_cycles
			cycles_cnt_bit_width	: integer := 12;

			-- Bram parameters
			N_bram			: integer := 58;
			N_weights_per_word	: integer := 7;

			-- Structure parameters
			N_inputs		: integer := 784;
			N_neurons		: integer := 400;

			-- Internal parameters
			shift			: integer := 10
		);

		port(

			-- Common signals ---------------------------------------------
			clk			: in std_logic;
			rst_n			: in std_logic;	

			-- Input interface signals ------------------------------------
			init_lfsr		: in std_logic;
			seed			: in unsigned(lfsr_bit_width-1 
							downto 0);

			-- Layer signals ----------------------------------------------  

			-- control input
			start			: in std_logic;	
			init_v_th		: in std_logic;

			-- address to select the neurons
			v_th_addr		: in std_logic_vector(
							neurons_addr_bit_width
							  downto 0);

			-- data input
			input_data		: in unsigned(N_inputs*
							lfsr_bit_width-1 
							downto 0);

			-- input parameters
			v_th_value		: in signed(neuron_bit_width-1 
							downto 0);		
			v_reset			: in signed(neuron_bit_width-1 
							downto 0);	
			inh_weight		: in signed(neuron_bit_width-1
							downto 0);		

			-- terminal counters 
			N_inputs_tc		: in std_logic_vector (
							inputs_addr_bit_width 
							downto 0);
			N_neurons_tc		: in std_logic_vector(
							neurons_addr_bit_width 
							downto 0);
			N_cycles_tc		: in std_logic_vector(
							cycles_cnt_bit_width-1
							downto 0);

			-- output
			ready			: out std_logic;
			cnt_out			: out std_logic_vector(N_neurons*
							cnt_out_bit_width-1
							downto 0);


			-- Memory signals --------------------------------------------- 
			-- input
			di		: in std_logic_vector(bram_word_length-1 
						downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(bram_addr_length-1 
						downto 0);
			bram_sel	: in std_logic_vector(bram_sel_length-1 
						downto 0)
		);

	end component spiker;


	component input_buffer is

		generic(
			addr_length	: integer := 10;
			data_bit_width	: integer := 8;
			N_data		: integer := 784
		);

		port(
			-- control input
			clk		: in std_logic;
			load_data	: in std_logic;
			data_addr	: in std_logic_vector(addr_length-1 
						downto 0);

			-- data input
			data_in		: in unsigned(data_bit_width-1 
						downto 0);

			-- data output
			data_out	: out unsigned(N_data*data_bit_width-1 
						downto 0)
		);

	end component input_buffer;


	component generic_mux is

		generic(
			N_sel		: integer := 8;
			bit_width	: integer := 8
		);

		port(
			-- input
			mux_in	: in std_logic_vector(2**N_sel*bit_width-1 
					downto 0);
			mux_sel	: in std_logic_vector(N_sel-1 downto 0);

			-- output
			mux_out	: out std_logic_vector(bit_width-1 downto 0)
		);

	end component generic_mux;



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


	component decoder is

		generic(
			N	: integer := 8		
		);

		port(
			-- input
			encoded_in	: in std_logic_vector(N-1 downto 0);

			-- output
			decoded_out	: out  std_logic_vector(2**N -1 
						downto 0)
		);

	end component decoder;



begin


	wren			<= enable(0);
	init_v_th		<= enable(1);
	load_v_reset		<= enable(2);
	load_inh_weight		<= enable(3);
	load_N_inputs_tc	<= enable(4);
	load_N_neurons_tc	<= enable(5);
	load_N_cycles_tc	<= enable(6);
	init_lfsr		<= enable(7);
	load_input_data		<= enable(8);


	seed			<= unsigned(data(lfsr_bit_width-1 downto 0));
	input_data		<= unsigned(data(input_data_bit_width-1
					downto 0));
	N_inputs_tc_value	<= data(inputs_addr_bit_width downto 0);
	N_neurons_tc_value	<= data(neurons_addr_bit_width downto 0);
	N_cycles_tc_value	<= data(cycles_cnt_bit_width-1 downto 0);
	v_th_value		<= signed(data(neuron_bit_width-1 downto 0));
	v_reset_value		<= signed(data(neuron_bit_width-1 downto 0));
	inh_weight_value	<= signed(data(neuron_bit_width-1 downto 0));
	di			<= data(bram_word_length-1 downto 0);

	cnt_out_sel		<= sel(neurons_addr_bit_width-1 downto 0);
	bram_sel		<= sel(bram_sel_length-1 downto 0);		

	wraddr			<= addr(bram_addr_length-1 downto 0);
	v_th_addr		<= addr(neurons_addr_bit_width downto 0);
	input_data_addr		<= addr(inputs_addr_bit_width-1 downto 0);



	data_padding	: process(buffered_data)
	begin

		pad	: for i in 0 to N_inputs - 1
		loop

			padded_buff_data((i+1)*lfsr_bit_width-1 downto
				(i+1)*input_data_bit_width) <= (others => '0');
			padded_buff_data((i+1)*input_data_bit_width-1 downto
				i*input_data_bit_width) <=
				buffered_data((i+1)*input_data_bit_width-1 
				downto i*input_data_bit_width);

		end loop pad;

	end process data_padding;
	


	load_decoder	: decoder

		generic map(
			N		=> load_bit_width
		)

		port map(
			-- input
			encoded_in	=> load,

			-- output
			decoded_out	=> enable
		);



	v_reset_reg	: reg_signed 

		generic map(
			-- parallelism
			N	=> neuron_bit_width
		)

		port map(	
			-- inputs	
			clk	=> clk,
			en	=> load_v_reset,
			reg_in	=> v_reset_value,

			-- outputs
			reg_out	=> v_reset
		);


	inh_weight_reg	: reg_signed 

		generic map(
			-- parallelism
			N	=> neuron_bit_width
		)

		port map(	
			-- inputs	
			clk	=> clk,
			en	=> load_inh_weight,
			reg_in	=> inh_weight_value,

			-- outputs
			reg_out	=> inh_weight
		);


	N_inputs_tc_reg	: reg 

		generic map(
			-- parallelism
			N	=> inputs_addr_bit_width + 1 
		)

		port map(	
			-- inputs	
			clk	=> clk,
			en	=> load_N_inputs_tc,
			reg_in	=> N_inputs_tc_value,

			-- outputs
			reg_out	=> N_inputs_tc
		);

	N_neurons_tc_reg	: reg 

		generic map(
			-- parallelism
			N	=> neurons_addr_bit_width + 1
		)

		port map(	
			-- Inputs	
			clk	=> clk,
			en	=> load_N_neurons_tc,
			reg_in	=> N_neurons_tc_value,

			-- outputs
			reg_out	=> N_neurons_tc
		);

	N_cycles_tc_reg	: reg 

		generic map(
			-- parallelism
			N	=> cycles_cnt_bit_width
		)

		port map(	
			-- Inputs	
			clk	=> clk,
			en	=> load_N_cycles_tc,
			reg_in	=> N_cycles_tc_value,

			-- outputs
			reg_out	=> N_cycles_tc
		);





	data_buffer	: input_buffer 

		generic map(
			addr_length	=> inputs_addr_bit_width,
			data_bit_width	=> input_data_bit_width,
			N_data		=> N_inputs
		)

		port map(
			-- control input
			clk		=> clk,	
			load_data	=> load_input_data,
			data_addr	=> input_data_addr,

			-- data input
			data_in		=> input_data,

			-- data output
			data_out	=> buffered_data
		);



	snn	: spiker 

		generic map(

			-- int parallelism
			neuron_bit_width	=> neuron_bit_width,
			weights_bit_width	=> weights_bit_width,
			bram_word_length	=> bram_word_length,
			bram_addr_length	=> bram_addr_length, 
			bram_sel_length		=> bram_sel_length, 
			bram_we_length		=> bram_we_length,
			input_data_bit_width	=> input_data_bit_width,
			lfsr_bit_width		=> lfsr_bit_width,
			cnt_out_bit_width	=> cnt_out_bit_width,

			-- Note: bit-width of the next three counters must be
			-- ceil(log_2(max_count)) + 1
			inputs_addr_bit_width	=> inputs_addr_bit_width,
			neurons_addr_bit_width	=> neurons_addr_bit_width,
			cycles_cnt_bit_width	=> cycles_cnt_bit_width,

			-- Number of block rams instantiated
			N_bram			=> N_bram,
			N_weights_per_word	=> N_weights_per_word,

			-- Structure parameters
			N_inputs		=> N_inputs,
			N_neurons		=> N_neurons,

			-- Internal parameters
			shift			=> shift
			)

		port map(

			-- Common signals --------------------------------------
			clk			=> clk,
			rst_n			=> rst_n,


			-- Input interface signals -----------------------------
			init_lfsr		=> init_lfsr,
			seed			=> seed,
						

			-- Layer signals ---------------------------------------

			-- control input
			start			=> start,
			init_v_th		=> init_v_th,

			-- address to select the neurons
			v_th_addr		=> v_th_addr,
                                                                   
			-- data input
			input_data		=> padded_buff_data,
                                                                   
			-- input parameters
			v_th_value		=> v_th_value,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
                                                                   
			-- terminal counters
			N_inputs_tc		=> N_inputs_tc,
			N_neurons_tc		=> N_neurons_tc,
			N_cycles_tc		=> N_cycles_tc,
                                                                   
			-- output
			ready			=> ready,
			cnt_out			=> counters(N_neurons*
							cnt_out_bit_width-1
							downto 0),
						
						


			-- Memory signals --------------------------------------
			-- input
			di			=> di,
			rden			=> rden,
			wren			=> wren,
			wraddr			=> wraddr,
			bram_sel		=> bram_sel
		);



	output_selector	: generic_mux 

		generic map(
			N_sel		=> neurons_addr_bit_width,
			bit_width	=> cnt_out_bit_width
		)

		port map(
			-- input
			mux_in		=> counters,
			mux_sel		=> cnt_out_sel,

			-- output
			mux_out		=> cnt_out
		);


end architecture behaviour;
