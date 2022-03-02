library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spiker is
end entity spiker;



architecture behaviour of spiker is

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


	component layer is

		generic(

			-- int parallelism
			parallelism		: integer := 16;
			weightsParallelism	: integer := 5;

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
							(N_neurons*weightsParallelism-1
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



begin

end architecture behaviour;
