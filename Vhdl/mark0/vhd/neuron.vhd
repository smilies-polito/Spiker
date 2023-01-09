library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron is

	generic(
		-- parallelism
		N		: integer := 8;

		-- shift amount
		shift		: integer := 1
	);

	port(
		-- input controls
		clk		: in std_logic;
		rst_n		: in std_logic;
		exp_exc_start	: in std_logic;
		rest_inh_start	: in std_logic;
		mask_neuron	: in std_logic;
		input_spike	: in std_logic;

		-- input parameters
		v_th_0		: in signed(N-1 downto 0);
		v_rest		: in signed(N-1 downto 0);
		v_reset		: in signed(N-1 downto 0);
		v_th_plus	: in signed(N-1 downto 0);
		inh_weight	: in signed(N-1 downto 0);
		exc_weight	: in signed(N-1 downto 0);

		-- output
		out_spike	: out std_logic
	);

end entity neuron;


architecture behaviour of neuron is


	-- from control unit towards datapath
	signal v_update_sel	: std_logic_vector(1 downto 0);
	signal v_or_v_th	: std_logic;
	signal add_or_sub	: std_logic;
	signal update_or_rest	: std_logic;
	signal v_reset_sel	: std_logic;
	signal v_th_en		: std_logic;
	signal v_en		: std_logic;

	-- from datapath towards control unit
	signal exceed_v_th	: std_logic;


	component neuron_datapath is

		generic(
			-- parallelism
			N		: integer := 8;

			-- shift amount
			shift		: integer := 1
		);

		port(
			-- input parameters
			v_th_0		: in signed(N-1 downto 0);
			v_rest		: in signed(N-1 downto 0);
			v_reset		: in signed(N-1 downto 0);
			v_th_plus	: in signed(N-1 downto 0);
			inh_weight	: in signed(N-1 downto 0);
			exc_weight	: in signed(N-1 downto 0);

			-- input controls
			clk		: in std_logic;
			v_update_sel	: in std_logic_vector(1 downto 0);
			v_or_v_th	: in std_logic;
			add_or_sub	: in std_logic;
			update_or_rest	: in std_logic;
			v_reset_sel	: in std_logic;
			v_th_en		: in std_logic;
			v_en		: in std_logic;
			
			-- output
			exceed_v_th	: out std_logic		
		);

	end component neuron_datapath;



	component neuron_cu is

		port(
			-- input
			clk		: in std_logic;
			rst_n		: in std_logic;
			exp_exc_start	: in std_logic;
			rest_inh_start	: in std_logic;
			mask_neuron	: in std_logic;
			input_spike	: in std_logic;

			-- input from datapath
			exceed_v_th	: in std_logic;

			-- control output
			v_update_sel	: out std_logic_vector(1 downto 0);
			v_or_v_th	: out std_logic;
			add_or_sub	: out std_logic;
			update_or_rest	: out std_logic;
			v_reset_sel	: out std_logic;
			v_th_en		: out std_logic;
			v_en		: out std_logic;

			-- output
			out_spike	: out std_logic
		);

	end component neuron_cu;


begin


	datapath : neuron_datapath
		generic map(
			-- parallelism
			N		=> N,

			-- shift amount
			shift		=> shift
		)

		port map(
			-- input parameters
			v_th_0		=> v_th_0,	   	
			v_rest		=> v_rest,	       
			v_reset		=> v_reset,	       
			v_th_plus	=> v_th_plus,      
			inh_weight	=> inh_weight,     
			exc_weight	=> exc_weight,    

			-- input controls
			clk		=> clk,
			v_update_sel	=> v_update_sel,  
			v_or_v_th	=> v_or_v_th,	     
			add_or_sub	=> add_or_sub,	     
			update_or_rest	=> update_or_rest,	     
			v_reset_sel	=> v_reset_sel,	     
			v_th_en		=> v_th_en,		     
			v_en		=> v_en,		     
			
			-- output
			exceed_v_th	=> exceed_v_th
		);


	control_unit : neuron_cu
		port map(
			-- input
			clk		=> clk,		
			rst_n		=> rst_n,		     
			exp_exc_start	=> exp_exc_start,	     
			rest_inh_start	=> rest_inh_start,	     
			mask_neuron	=> mask_neuron,	     
			input_spike	=> input_spike,	     

			-- input from datapath
			exceed_v_th	=> exceed_v_th,

			-- control output
			v_update_sel	=> v_update_sel,		
			v_or_v_th	=> v_or_v_th,	     
			add_or_sub	=> add_or_sub,	     
			update_or_rest	=> update_or_rest,	     
			v_reset_sel	=> v_reset_sel,	
			v_th_en		=> v_th_en,	     
			v_en		=> v_en,
                                                           
			-- output
			out_spike	=> out_spike	     
		);



end architecture behaviour;
