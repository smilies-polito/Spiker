library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron is

	generic(
		-- bit-width
		neuron_bit_width	: integer := 16;
		weights_bit_width	: integer := 5;

		-- shift amount
		shift			: integer := 10
	);

	port(
		-- input controls
		clk			: in std_logic;
		rst_n			: in std_logic;
		start			: in std_logic;
		stop			: in std_logic;
		exc_or			: in std_logic;
		exc_stop		: in std_logic;
		inh_or			: in std_logic;
		inh_stop		: in std_logic;
		input_spike		: in std_logic;

		-- to load the threshold
		v_th_en			: in std_logic;

		-- input parameters
		v_th_value		: in signed(neuron_bit_width-1 
						downto 0);
		v_reset			: in signed(neuron_bit_width-1
						downto 0);
		inh_weight		: in signed(neuron_bit_width-1
						downto 0);
		exc_weight		: in signed(weights_bit_width-1
						downto 0);

		-- output
		out_spike		: out std_logic;
		neuron_ready		: out std_logic
	);

end entity neuron;


architecture behaviour of neuron is


	-- from control unit towards datapath
	signal update_sel	: std_logic_vector(1 downto 0);
	signal add_or_sub	: std_logic;
	signal v_update		: std_logic;
	signal v_en		: std_logic;
	signal v_rst_n		: std_logic;

	-- from datapath towards control unit
	signal exceed_v_th	: std_logic;


	component neuron_datapath is

		generic(
			-- parallelism
			neuron_bit_width		: integer := 16;
			weights_bit_width		: integer := 5;

			-- shift amount
			shift				: integer := 1
		);

		port(
			-- input parameters
			v_th_value		: in signed(neuron_bit_width-1 
							downto 0);
			v_reset			: in signed(neuron_bit_width-1 
							downto 0);
			inh_weight		: in signed(neuron_bit_width-1
							downto 0);
			exc_weight		: in signed(weights_bit_width-1
							downto 0);

			-- input controls
			clk			: in std_logic;
			update_sel		: in std_logic_vector(1 
							downto 0);
			add_or_sub		: in std_logic;
			v_update		: in std_logic;
			v_th_en			: in std_logic;
			v_en			: in std_logic;
			v_rst_n			: in std_logic;
			
			-- output
			exceed_v_th		: out std_logic		
		);

	end component neuron_datapath;


	component neuron_cu is

		port(
			-- input
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;
			exc_or		: in std_logic;
			exc_stop	: in std_logic;
			inh_or		: in std_logic;
			inh_stop	: in std_logic;
			input_spike	: in std_logic;

			-- input from datapath
			exceed_v_th	: in std_logic;

			-- control output
			update_sel	: out std_logic_vector(1 downto 0);
			add_or_sub	: out std_logic;
			v_update	: out std_logic;
			v_en		: out std_logic;
			v_rst_n		: out std_logic;

			-- output
			out_spike	: out std_logic;
			neuron_ready	: out std_logic
		);

	end component neuron_cu;




begin


	datapath : neuron_datapath
		generic map(
			-- parallelism
			neuron_bit_width	=> neuron_bit_width,
			weights_bit_width	=> weights_bit_width,

			-- shift amount
			shift			=> shift
		)

		port map(
			-- input parameters
			v_th_value		=> v_th_value,	   	
			v_reset			=> v_reset,	       
			inh_weight		=> inh_weight,     
			exc_weight		=> exc_weight,    

			-- input controls
			clk			=> clk,
			update_sel		=> update_sel,		
			add_or_sub		=> add_or_sub,	     
			v_update		=> v_update,	     
			v_th_en			=> v_th_en,	     
			v_en			=> v_en,
			v_rst_n			=> v_rst_n,
     
			-- output
			exceed_v_th		=> exceed_v_th
		);


	control_unit : neuron_cu
		port map(
			-- input
			clk		=> clk,		
			rst_n		=> rst_n,
   			start		=> start,			
			stop		=> stop,	     
			exc_or		=> exc_or,	     
			exc_stop	=> exc_stop,	     
			inh_or		=> inh_or,	     
			inh_stop	=> inh_stop,	     
			input_spike	=> input_spike,	     

			-- input from datapath
			exceed_v_th	=> exceed_v_th,

			-- control output
			update_sel	=> update_sel,		
			add_or_sub	=> add_or_sub,	     
			v_update	=> v_update,	     
			v_en		=> v_en,
			v_rst_n		=> v_rst_n,
                                                           
			-- output
			out_spike	=> out_spike,
	   		neuron_ready	=> neuron_ready		
		);



end architecture behaviour;
