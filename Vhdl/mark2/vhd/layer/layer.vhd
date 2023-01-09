library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity layer is

	generic(

		-- internal parallelism
		parallelism		: integer := 16;

		-- excitatory spikes
		input_parallelism	: integer := 8;
		N_exc_cnt		: integer := 3;

		-- inhibitory spikes
		layer_size		: integer := 4;
		N_inh_cnt		: integer := 2;

		-- elaboration steps
		N_cycles_cnt		: integer := 4;

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
		v_th_0			: in signed(parallelism-1 downto 0);		
		v_reset			: in signed(parallelism-1 downto 0);	
		inh_weight		: in signed(parallelism-1 downto 0);		
		v_th_plus		: in signed(parallelism-1 downto 0);	
		exc_weights		: in signed
					(layer_size*parallelism-1 downto 0);

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

end entity layer;


architecture behaviour of layer is


	-- from datapath towards control unit
	signal exc_or			: std_logic;		
	signal exc_stop			: std_logic;		
	signal inh_or			: std_logic;		
	signal inh_stop			: std_logic;
	signal stop			: std_logic;		

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
	signal cycles_cnt_rst_n		: std_logic;	
	signal cycles_cnt_en		: std_logic;

	-- intermediate signals
	signal neurons_ready		: std_logic;
	signal layer_ready		: std_logic;




	component layer_datapath is

		generic(

			-- internal parallelism
			parallelism		: integer := 16;

			-- excitatory spikes
			input_parallelism	: integer := 8;
			N_exc_cnt		: integer := 3;

			-- inhibitory spikes
			layer_size		: integer := 4;
			N_inh_cnt		: integer := 2;

			-- elaboration steps
			N_cycles_cnt		: integer := 4;

			-- exponential decay shift
			shift			: integer := 1
				
		);

		port(

			-- input from the output world
			clk			: in std_logic;
			rst_n			: in std_logic;	
			start			: in std_logic;	
			input_spikes		: in std_logic_vector
							(input_parallelism-1 downto 0);

			-- control input
			exc_en			: in std_logic;	
			anticipate_exc		: in std_logic;	
			inh_en			: in std_logic;	
			anticipate_inh		: in std_logic;	
			exc_cnt_rst_n		: in std_logic;	
			exc_cnt_en		: in std_logic;	
			inh_cnt_rst_n		: in std_logic;	
			inh_cnt_en		: in std_logic;	
			exc_or_inh_sel		: in std_logic;	
			inh			: in std_logic;	
			cycles_cnt_rst_n	: in std_logic;	
			cycles_cnt_en		: in std_logic;	

			-- input parameters
			v_th_0			: in signed(parallelism-1 downto 0);		
			v_reset			: in signed(parallelism-1 downto 0);	
			inh_weight		: in signed(parallelism-1 downto 0);		
			v_th_plus		: in signed(parallelism-1 downto 0);	
			exc_weights		: in signed
						(layer_size*parallelism-1 downto 0);

			-- number of inputs, neurons and cycles
			N_inputs		: in std_logic_vector
							(N_exc_cnt-1 downto 0);
			N_neurons		: in std_logic_vector
							(N_inh_cnt-1 downto 0);
			N_cycles		: in std_logic_vector
							(N_cycles_cnt-1 downto 0);

			-- control output
			exc_or			: out std_logic;
			exc_stop		: out std_logic;
			inh_or			: out std_logic;
			inh_stop		: out std_logic;
			stop			: out std_logic;

			-- output
			out_spikes		: out std_logic_vector
							(layer_size-1 downto 0);
			neurons_ready		: out std_logic;
			exc_cnt			: out std_logic_vector
							(N_exc_cnt-1 downto 0)
		
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
			exc_cnt_rst_n		: out std_logic;	
			exc_cnt_en		: out std_logic;	
			inh_cnt_rst_n		: out std_logic;	
			inh_cnt_en		: out std_logic;	
			exc_or_inh_sel		: out std_logic;	
			inh			: out std_logic;	
			cycles_cnt_rst_n	: out std_logic;	
			cycles_cnt_en		: out std_logic;	

			-- output
			sample			: out std_logic;
			layer_ready		: out std_logic
		);

	end component layer_cu;


begin

	ready	<= neurons_ready and layer_ready;


	datapath	: layer_datapath

		generic map(

			-- internal parallelism
			parallelism		=> parallelism,
                                                                   
			-- excitatory spikes       
			input_parallelism	=> input_parallelism,
			N_exc_cnt		=> N_exc_cnt,
                                                                   
			-- inhibitory spikes      
			layer_size		=> layer_size,
			N_inh_cnt		=> N_inh_cnt,
                                                                   
			-- elaboration steps     
			N_cycles_cnt		=> N_cycles_cnt,
                                                                   
			-- exponential decay shift
			shift			=> shift
		)

		port map(

			-- input from the output world
			clk			=> clk,
			rst_n			=> rst_n,
			start			=> start,			
			input_spikes		=> input_spikes,
	
                                                                   
			-- control input           
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
			cycles_cnt_rst_n	=> cycles_cnt_rst_n,
			cycles_cnt_en		=> cycles_cnt_en,
                                                                   
			-- input parameters        
			v_th_0			=> v_th_0,
			v_reset			=> v_reset,
			inh_weight		=> inh_weight,
			v_th_plus		=> v_th_plus,
			exc_weights		=> exc_weights,
						   		
                                                                   
			-- number of inputs, neuron
			N_inputs		=> N_inputs,
			N_neurons		=> N_neurons,		
			N_cycles		=> N_cycles,
                                                                   
			-- control output          
			exc_or			=> exc_or,
			exc_stop		=> exc_stop,
			inh_or			=> inh_or,
			inh_stop		=> inh_stop,
			stop			=> stop,
                                                                   
			-- output                  
			out_spikes		=> out_spikes,     
			neurons_ready		=> neurons_ready,
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
			cycles_cnt_rst_n	=> cycles_cnt_rst_n,
			cycles_cnt_en		=> cycles_cnt_en,
                                                                   
			-- output                  -- output
			sample			=> sample,
			layer_ready		=> layer_ready
		);




end architecture behaviour;
