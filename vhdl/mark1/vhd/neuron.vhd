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
		start		: in std_logic;
		start1		: in std_logic;
		start2		: in std_logic;
		rest_en		: in std_logic;
		mask_neuron	: in std_logic;
		input_spike	: in std_logic;

		-- input parameters
		v_th_0		: in signed(N-1 downto 0);
		v_reset		: in signed(N-1 downto 0);
		inh_weight	: in signed(N-1 downto 0);
		exc_weight	: in signed(N-1 downto 0);
		v_th_plus	: in signed(N-1 downto 0);

		-- output
		out_spike	: out std_logic;
		neuron_ready	: out std_logic
	);

end entity neuron;


architecture behaviour of neuron is


	-- from control unit towards datapath
	signal update_sel	: std_logic_vector(1 downto 0);
	signal v_or_v_th	: std_logic;
	signal add_or_sub	: std_logic;
	signal v_th_update	: std_logic;
	signal v_update		: std_logic;
	signal v_th_en		: std_logic;
	signal v_en		: std_logic;
	signal v_rst_n		: std_logic;

	-- from datapath towards control unit
	signal exceed_v_th	: std_logic;


	component neuron_datapath is

		generic(
			-- parallelism
			N			: integer := 8;

			-- shift amount
			shift			: integer := 1
		);

		port(
			-- input parameters
			v_th_0		: in signed(N-1 downto 0);
			v_reset		: in signed(N-1 downto 0);
			inh_weight	: in signed(N-1 downto 0);
			exc_weight	: in signed(N-1 downto 0);
			v_th_plus	: in signed(N-1 downto 0);

			-- input controls
			clk		: in std_logic;
			update_sel	: in std_logic_vector(1 downto 0);
			v_or_v_th	: in std_logic;
			add_or_sub	: in std_logic;
			v_th_update	: in std_logic;
			v_update	: in std_logic;
			v_th_en		: in std_logic;
			v_en		: in std_logic;
			v_rst_n		: in std_logic;
			
			-- output
			exceed_v_th	: out std_logic		
		);

	end component neuron_datapath;



	component neuron_cu is

		port(
			-- input
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			start1		: in std_logic;
			start2		: in std_logic;
			rest_en		: in std_logic;
			mask_neuron	: in std_logic;
			input_spike	: in std_logic;

			-- input from datapath
			exceed_v_th	: in std_logic;

			-- control output
			update_sel	: out std_logic_vector(1 downto 0);
			v_or_v_th	: out std_logic;
			add_or_sub	: out std_logic;
			v_th_update	: out std_logic;
			v_update	: out std_logic;
			v_th_en		: out std_logic;
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
			N		=> N,

			-- shift amount
			shift		=> shift
		)

		port map(
			-- input parameters
			v_th_0		=> v_th_0,	   	
			v_reset		=> v_reset,	       
			inh_weight	=> inh_weight,     
			exc_weight	=> exc_weight,    
			v_th_plus	=> v_th_plus,      

			-- input controls
			clk		=> clk,
			update_sel	=> update_sel,		
			v_or_v_th	=> v_or_v_th,	     
			add_or_sub	=> add_or_sub,	     
			v_update	=> v_update,	     
			v_th_update	=> v_th_update,	     
			v_th_en		=> v_th_en,	     
			v_en		=> v_en,
			v_rst_n		=> v_rst_n,
     
			-- output
			exceed_v_th	=> exceed_v_th
		);


	control_unit : neuron_cu
		port map(
			-- input
			clk		=> clk,		
			rst_n		=> rst_n,
   			start		=> start,			
			start1		=> start1,	     
			start2		=> start2,	     
			rest_en		=> rest_en,	     
			mask_neuron	=> mask_neuron,	     
			input_spike	=> input_spike,	     

			-- input from datapath
			exceed_v_th	=> exceed_v_th,

			-- control output
			update_sel	=> update_sel,		
			v_or_v_th	=> v_or_v_th,	     
			add_or_sub	=> add_or_sub,	     
			v_update	=> v_update,	     
			v_th_update	=> v_th_update,	     
			v_th_en		=> v_th_en,	     
			v_en		=> v_en,
			v_rst_n		=> v_rst_n,
                                                           
			-- output
			out_spike	=> out_spike,
	   		neuron_ready	=> neuron_ready		
		);



end architecture behaviour;
