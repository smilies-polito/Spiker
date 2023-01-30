library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity layer is

	generic(

		-- int parallelism
		neuron_bit_width	: integer := 16;
		weights_bit_width	: integer := 5;

		-- input spikes
		N_inputs		: integer := 392; --784;

		-- must be one bit larger that the bit-width required to count
		-- up to N_inputs
		inputs_cnt_bit_width	: integer := 11;

		-- inhibitory spikes
		N_neurons		: integer := 400;

		-- must be one bit larger that the bit-width required to count
		-- up to N_neurons
		neurons_cnt_bit_width	: integer := 10;

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
						neurons_cnt_bit_width-1
						downto 0);

		-- data input
		input_spikes		: in std_logic_vector
						(N_inputs-1 downto 0);

		-- input parameters
		v_th_value		: in signed(neuron_bit_width-1 
						downto 0);		
		v_reset			: in signed(neuron_bit_width-1 
						downto 0);	
		inh_weight		: in signed(neuron_bit_width-1 
						downto 0);
		exc_weights		: in signed
						(N_neurons*weights_bit_width-1
						 downto 0);

		-- terminal counters 
		N_inputs_tc		: in std_logic_vector
						(inputs_cnt_bit_width-1 
						downto 0);
		N_neurons_tc		: in std_logic_vector
						(neurons_cnt_bit_width-1 
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
						(inputs_cnt_bit_width-1
						downto 0)
	);

end entity layer;


architecture behaviour of layer is


	-- from datapath towards control unit
	signal exc_or			: std_logic;		
	signal exc_stop			: std_logic;		
	signal inh_or			: std_logic;		
	signal inh_stop			: std_logic;

	-- from control unit towards datapath
	signal exc_en			: std_logic;	
	signal anticipate_exc		: std_logic;	
	signal inh_en			: std_logic;	
	signal anticipate_inh		: std_logic;	
	signal exc_cnt_rst_n		: std_logic;	
	signal exc_cnt_en		: std_logic;	
	signal inh_cnt_rst_n		: std_logic;	
	signal inh_cnt_en		: std_logic;	
	signal exc_or_inh_sel		: std_logic;	
	signal inh			: std_logic;	

	-- intermediate signals
	signal all_ready		: std_logic;
	signal layer_ready		: std_logic;




	component layer_datapath is

		generic(

			-- int parallelism
			neuron_bit_width	: integer := 16;
			weights_bit_width	: integer := 5;

			-- input spikes
			N_inputs		: integer := 784;

			-- must be one bit larger that the parallelism required
			-- to count up to N_inputs
			inputs_cnt_bit_width	: integer := 11;

			-- inhibitory spikes
			N_neurons		: integer := 400;

			-- must be one bit larger that the parallelism required
			-- to count up to N_neurons
			neurons_cnt_bit_width	: integer := 10;

			-- exponential decay shift
			shift			: integer := 10
				
		);

		port(
			-- control input
			clk			: in std_logic;
			exc_en			: in std_logic;	
			anticipate_exc		: in std_logic;	
			inh_en			: in std_logic;	
			anticipate_inh		: in std_logic;	
			exc_cnt_en		: in std_logic;	
			exc_cnt_rst_n		: in std_logic;	
			inh_cnt_en		: in std_logic;	
			inh_cnt_rst_n		: in std_logic;	
			exc_or_inh_sel		: in std_logic;	
			init_v_th		: in std_logic;
			rst_n			: in std_logic;	
			start			: in std_logic;	
			stop			: in std_logic;	
			inh			: in std_logic;	

			-- address to select the neurons
			v_th_addr		: in std_logic_vector
							(neurons_cnt_bit_width-1
							downto 0);

			-- data input
			input_spikes		: in std_logic_vector
							(N_inputs-1 downto 0);

			-- input parameters
			v_th_value		: in signed(neuron_bit_width-1 
							downto 0);		
			v_reset			: in signed(neuron_bit_width-1 
							downto 0);	
			inh_weight		: in signed(neuron_bit_width-1 
							downto 0);		
			exc_weights		: in signed
							(N_neurons*
							 weights_bit_width-1
							 downto 0);

			-- terminal counters 
			N_inputs_tc		: in std_logic_vector
							(inputs_cnt_bit_width-1
							 downto 0);
			N_neurons_tc		: in std_logic_vector
							(neurons_cnt_bit_width-1
							 downto 0);

			-- control output
			exc_or			: out std_logic;
			exc_stop		: out std_logic;
			inh_or			: out std_logic;
			inh_stop		: out std_logic;

			-- output
			out_spikes		: out std_logic_vector
							(N_neurons-1 downto 0);
			all_ready		: out std_logic;

			-- output address to select the excitatory weights
			exc_cnt			: out std_logic_vector
							(inputs_cnt_bit_width-1 
							 downto 0)
		
		);

	end component layer_datapath;




	component layer_cu is

		port(
			-- input
			clk			: in std_logic;
			rst_n			: in std_logic;
			start			: in std_logic;

			-- signals from datapath
			exc_or			: in std_logic;		
			exc_stop		: in std_logic;		
			inh_or			: in std_logic;		
			inh_stop		: in std_logic;
			stop			: in std_logic;		

			-- towards datapath
			exc_en			: out std_logic;	
			anticipate_exc		: out std_logic;	
			inh_en			: out std_logic;	
			anticipate_inh		: out std_logic;	
			exc_cnt_en		: out std_logic;	
			exc_cnt_rst_n		: out std_logic;	
			inh_cnt_en		: out std_logic;	
			inh_cnt_rst_n		: out std_logic;	
			exc_or_inh_sel		: out std_logic;	
			inh			: out std_logic;	

			-- output
			cycles_cnt_rst_n	: out std_logic;	
			cycles_cnt_en		: out std_logic;	
			sample			: out std_logic;
			layer_ready		: out std_logic
		);

	end component layer_cu;

begin

	ready	<= all_ready and layer_ready;


	datapath	: layer_datapath

		generic map(

			-- internal parallelism
			neuron_bit_width	=> neuron_bit_width,
			weights_bit_width	=> weights_bit_width,
                                                                   
			-- input spikes       
			N_inputs		=> N_inputs,

			-- must be one bit larger that the parallelism required
			-- to count up to N_inputs
			inputs_cnt_bit_width	=> inputs_cnt_bit_width,
                                                                   
			-- inhibitory spikes      
			N_neurons		=> N_neurons,

			-- must be one bit larger that the parallelism required
			-- to count up to N_neurons
			neurons_cnt_bit_width	=> neurons_cnt_bit_width,
                                                                   
			-- exponential decay shift
			shift			=> shift
		)

		port map(

			-- input from the output world
			clk			=> clk,
			exc_en			=> exc_en,
			anticipate_exc		=> anticipate_exc,
			inh_en			=> inh_en,
			anticipate_inh		=> anticipate_inh,
			exc_cnt_rst_n		=> exc_cnt_rst_n,
			exc_cnt_en		=> exc_cnt_en,
			inh_cnt_rst_n		=> inh_cnt_rst_n,
			inh_cnt_en		=> inh_cnt_en,
			exc_or_inh_sel		=> exc_or_inh_sel,
			init_v_th		=> init_v_th,
			rst_n			=> rst_n,
			start			=> start,
			stop			=> stop,
			inh			=> inh,

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
                                                                   
			-- control output          
			exc_or			=> exc_or,
			exc_stop		=> exc_stop,
			inh_or			=> inh_or,
			inh_stop		=> inh_stop,
                                                                   
			-- output                  
			out_spikes		=> out_spikes,     
			all_ready		=> all_ready,
			exc_cnt			=> exc_cnt
		);


	control_unit	: layer_cu

		port map(
			-- input
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,
                                                                   
			-- signals from datapath
			exc_or			=> exc_or,
			exc_stop		=> exc_stop,
			inh_or			=> inh_or,
			inh_stop		=> inh_stop,
			stop			=> stop,
                                                                   
			-- towards datapath       
			exc_en			=> exc_en,
			anticipate_exc		=> anticipate_exc,
			inh_en			=> inh_en,
			anticipate_inh		=> anticipate_inh,
			exc_cnt_rst_n		=> exc_cnt_rst_n,
			exc_cnt_en		=> exc_cnt_en,
			inh_cnt_rst_n		=> inh_cnt_rst_n,
			inh_cnt_en		=> inh_cnt_en,
			exc_or_inh_sel		=> exc_or_inh_sel,
			inh			=> inh,
                                                                   
			-- output 
			cycles_cnt_rst_n	=> cycles_cnt_rst_n,
			cycles_cnt_en		=> cycles_cnt_en,
			sample			=> sample,
			layer_ready		=> layer_ready
		);




end architecture behaviour;
