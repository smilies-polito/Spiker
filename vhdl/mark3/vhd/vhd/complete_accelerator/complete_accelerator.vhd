library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity complete_accelerator is

	generic(
		-- int parallelism
		parallelism		: integer := 16;
		weights_bit_width	: integer := 5;

		-- input buffer
		input_data_addr_length	: integer := 10;


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
		N_out			: integer := 16;
		N_cnt_out_sel		: integer := 10
	);

	port(

		-- Common signals ---------------------------------------------
		clk			: in std_logic;
		rst_n			: in std_logic;	

		-- Input buffer signals -----------------------------
		input_data_addr		: in std_logic_vector(
						input_data_addr_length-1
						downto 0);
			

		-- Input interface signals ------------------------------------
		init_lfsr		: in std_logic;
		seed			: in unsigned(lfsr_bit_width-1 
						downto 0);

		-- Layer signals ----------------------------------------------  

		-- control input
		start			: in std_logic;	
		init_v_th		: in std_logic;
		load_input_data		: in std_logic;
		load_v_reset		: in std_logic;
		load_inh_weight		: in std_logic;

		-- address to select the neurons
		v_th_addr		: in std_logic_vector(N_neurons_cnt-1
						  downto 0);

		-- data input
		input_data		: in unsigned(input_data_bit_width-1 
						downto 0);

		-- input parameters
		v_th_value		: in signed(parallelism-1 downto 0);		
		v_reset_value		: in signed(parallelism-1 downto 0);	
		inh_weight_value	: in signed(parallelism-1 downto 0);		

		-- terminal counters 
		N_inputs_tc		: in std_logic_vector
						(N_inputs_cnt-1 downto 0);
		N_neurons_tc		: in std_logic_vector
						(N_neurons_cnt-1 downto 0);
		N_cycles_tc		: in std_logic_vector(N_cycles_cnt-1
						downto 0);
		
		-- output counters selector
		cnt_out_sel		: in std_logic_vector(N_cnt_out_sel-1
						downto 0);

		-- output
		ready			: out std_logic;
		cnt_out			: out std_logic_vector(N_out-1
						downto 0);


		-- Memory signals --------------------------------------------- 
		-- input
		di		: in std_logic_vector(35 downto 0);
		rden		: in std_logic;
		wren		: in std_logic;
		wraddr		: in std_logic_vector(9 downto 0);
		bram_sel	: in std_logic_vector(5 downto 0)
	);

end entity complete_accelerator;



architecture behaviour of complete_accelerator is

	signal buffered_data	: unsigned(N_inputs*input_data_bit_width-1 
					downto 0);
	signal padded_buff_data	: unsigned(N_inputs*lfsr_bit_width-1 
					downto 0);
	signal counters		: std_logic_vector(2**N_cnt_out_sel*N_out-1
						downto 0);
	signal v_reset		: signed(parallelism-1 downto 0);	
	signal inh_weight	: signed(parallelism-1 downto 0);		


	component spiker is

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

			-- Input interface bit width
			lfsr_bit_width		: integer := 16;

			-- Output counters parallelism
			N_out			: integer := 16
		);

		port(

			-- Common signals --------------------------------------
			clk			: in std_logic;
			rst_n			: in std_logic;	


			-- Input interface signals -----------------------------
			init_lfsr		: in std_logic;
			seed			: in unsigned(lfsr_bit_width-1
							downto 0);

			-- Layer signals ---------------------------------------

			-- control input
			start			: in std_logic;	
			init_v_th		: in std_logic;

			-- address to select the neurons
			v_th_addr		: in std_logic_vector(
							N_neurons_cnt-1 
							downto 0);

			-- data input
			input_data		: in unsigned(N_inputs*
							lfsr_bit_width-1 
							downto 0);

			-- input parameters
			v_th_value		: in signed(parallelism-1 
							downto 0);		
			v_reset			: in signed(parallelism-1
							downto 0);	
			inh_weight		: in signed(parallelism-1
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
			cnt_out			: out std_logic_vector(
							N_neurons*N_out-1
							downto 0);


			-- Memory signals --------------------------------------
			-- input
			di		: in std_logic_vector(35 downto 0);
			rden		: in std_logic;
			wren		: in std_logic;
			wraddr		: in std_logic_vector(9 downto 0);
			bram_sel	: in std_logic_vector(5 downto 0)
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
			data_addr	: in std_logic_vector(addr_length-1 downto 0);

			-- data input
			data_in		: in unsigned(data_bit_width-1 downto 0);

			-- data output
			data_out	: out unsigned(N_data*data_bit_width-1 downto 0)
		);

	end component input_buffer;


	component generic_mux is

		generic(
			N_sel		: integer := 8;
			bit_width	: integer := 8
		);

		port(
			-- input
			mux_in	: in std_logic_vector(2**N_sel*bit_width-1 downto 0);
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



begin

	data_padding	: process(buffered_data)
	begin

		pad	: for i in 0 to N_inputs - 1
		loop

			padded_buff_data((i+1)*lfsr_bit_width-1 downto
				(i+1)*input_data_bit_width) <= (others => '0');
			padded_buff_data((i+1)*input_data_bit_width-1 downto
				i*input_data_bit_width) <=
				buffered_data((i+1)*input_data_bit_width-1 downto
				i*input_data_bit_width);

		end loop pad;

	end process data_padding;

	v_reset_reg	: reg_signed 

		generic map(
			-- parallelism
			N	=> parallelism
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
			N	=> parallelism
		)

		port map(	
			-- inputs	
			clk	=> clk,
			en	=> load_inh_weight,
			reg_in	=> inh_weight_value,

			-- outputs
			reg_out	=> inh_weight
		);


	data_buffer	: input_buffer 

		generic map(
			addr_length	=> input_data_addr_length,
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
			parallelism		=> parallelism,
			weights_bit_width	=> weights_bit_width,
                                                                           
			-- memory parameters
			word_length		=> word_length,
			rdwr_addr_length	=> rdwr_addr_length,
			we_length		=> we_length,
			N_bram			=> N_bram,
			bram_addr_length	=> bram_addr_length,
                                                                           
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

			-- Input interface bit width
			lfsr_bit_width		=> lfsr_bit_width,

			-- Output counters parallelism
			N_out			=> N_out
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
			cnt_out			=> counters(N_neurons*N_out-1
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
			N_sel		=> N_cnt_out_sel,
			bit_width	=> N_out
		)

		port map(
			-- input
			mux_in		=> counters,
			mux_sel		=> cnt_out_sel,

			-- output
			mux_out		=> cnt_out
		);


end architecture behaviour;
