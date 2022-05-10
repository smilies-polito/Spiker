library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

library UNISIM;
use UNISIM.vcomponents.all;

library UNIMACRO;
use UNIMACRO.vcomponents.all;


entity neuron_and_bram_tb is
end entity neuron_and_bram_tb; 

architecture test of neuron_and_bram_tb is

	-- parallelism
	constant N			: integer := 16;
	constant N_weight		: integer := 5;

	-- shift amount
	constant shift			: integer := 10;
	constant v_reset_int		: integer := 5*2**3; 	  
	constant inh_weight_int		: integer := -15*2**3;
	constant v_th_value_int		: integer :=  589;

	-- common signals
	signal clk			: std_logic;

	-- memory signals
	signal mem_out			: std_logic_vector(35 downto 0); 	
	signal mem_in			: std_logic_vector(35 downto 0);	
	signal rdaddr			: std_logic_vector(9 downto 0);		
	signal rden			: std_logic;		
	signal regce			: std_logic;		
	signal rst			: std_logic;		
	signal we			: std_logic_vector(3 downto 0);		
	signal wraddr			: std_logic_vector(9 downto 0);		
	signal wren			: std_logic;		
	
	-- input controls
	signal rst_n			: std_logic;
	signal start			: std_logic;
	signal stop			: std_logic;
	signal exc_or			: std_logic;
	signal exc_stop			: std_logic;
	signal inh_or			: std_logic;
	signal inh_stop			: std_logic;
	signal input_spike		: std_logic;
	signal v_th_en			: std_logic;

	-- input parameters
	signal v_th_value		: signed(N-1 downto 0);
	signal v_reset			: signed(N-1 downto 0);
	signal inh_weight		: signed(N-1 downto 0);
	signal exc_weight		: signed(N_weight-1 downto 0);

	-- output
	signal out_spike		: std_logic;
	signal neuron_ready		: std_logic;

	-- debug output
	signal v_out			: signed(N-1 downto 0);


	component debug_neuron is

		generic(
			-- parallelism
			N		: integer := 16;
			N_weight	: integer := 5;

			-- shift amount
			shift		: integer := 1
		);

		port(
			-- input controls
			clk		: in std_logic;
			rst_n		: in std_logic;
			start		: in std_logic;
			stop		: in std_logic;
			exc_or		: in std_logic;
			exc_stop	: in std_logic;
			inh_or		: in std_logic;
			inh_stop	: in std_logic;
			input_spike	: in std_logic;

			-- to load the threshold
			v_th_en		: in std_logic;

			-- input parameters
			v_th_value	: in signed(N-1 downto 0);
			v_reset		: in signed(N-1 downto 0);
			inh_weight	: in signed(N-1 downto 0);
			exc_weight	: in signed(N_weight-1 downto 0);

			-- output
			out_spike	: out std_logic;
			neuron_ready	: out std_logic;

			-- debug output
			v_out		: out signed(N-1 downto 0)
		);



	end component debug_neuron;


begin

	-- connect memory and neuron
	exc_weight <= signed(mem_out(exc_weight'length-1 downto 0));

	-- initialize threshold
	v_th_value <= to_signed(v_th_value_int, v_th_value'length);


	neuron	: debug_neuron 

		generic map(
			-- parallelism
			N		=> N,
			N_weight	=> N_weight,

			-- shift amount
			shift		=> shift
		)

		port map(
			-- input controls
			clk		=> clk,
			rst_n		=> rst_n,
			start		=> start,
			stop		=> stop,
			exc_or		=> exc_or,
			exc_stop	=> exc_stop,
			inh_or		=> inh_or,
			inh_stop	=> inh_stop,
			input_spike	=> input_spike,

			-- to load the threshold
			v_th_en		=> v_th_en,

			-- input parameters
			v_th_value	=> v_th_value,
			v_reset		=> v_reset,
			inh_weight	=> inh_weight,
			exc_weight	=> exc_weight,

			-- output
			out_spike	=> out_spike,
			neuron_ready	=> neuron_ready,

			-- debug output
			v_out		=> v_out
		);



	BRAM_SDP_MACRO_inst : BRAM_SDP_MACRO
		generic map (

			-- Target BRAM, "18Kb" or "36Kb" 
			BRAM_SIZE 		=> "36Kb", 

			-- Target device: "VIRTEX5", "VIRTEX6", "7SERIES",
			-- "SPARTAN6" 
			DEVICE 			=> "7SERIES", 

			-- Valid values are 1-72 (37-72 only valid when
			-- BRAM_SIZE="36Kb")
			WRITE_WIDTH 		=> 36,

			-- Valid values are 1-72 (37-72 only valid when
			-- BRAM_SIZE="36Kb")
			READ_WIDTH 		=> 36,     

			-- Optional output register (0 or 1)
			DO_REG 			=> 0, 
			INIT_FILE 		=> "NONE",

			-- Collision check enable "ALL", "WARNING_ONLY",
			-- "GENERATE_X_ONLY" or "NONE" 
			SIM_COLLISION_CHECK 	=> "ALL", 
			
			--  Set/Reset value for port output
			SRVAL 			=> X"000000000", 

			-- Specify "READ_FIRST" for same clock or synchronous
			-- clocks. Specify "WRITE_FIRST for asynchrononous
			-- clocks on ports
			WRITE_MODE 		=> "READ_FIRST", 
						   
			--  Initial values on output port
			INIT 			=> X"000000000" 
		)


		port map (
			
			-- Output read data port, width defined by READ_WIDTH
			-- parameter
			do 	=> mem_out,         

			-- Input write data port, width defined by WRITE_WIDTH
			-- parameter
			di 	=> mem_in,         

			-- Input read address, width defined by read port depth
			rdaddr 	=> rdaddr, 

			-- 1-bit input read clock
			rdclk 	=> clk,   

			-- 1-bit input read port enable
			rden 	=> rden,     

			-- 1-bit input read output register enable
			regce 	=> '0',   

			-- 1-bit input reset 
			rst 	=> rst, 

			-- Input write enable, width defined by write port depth
			we 	=> we,         

			-- Input write address, width defined by write port
			-- depth
			wraddr 	=> wraddr, 

			-- 1-bit input write clock
			wrclk 	=> clk,   

			-- 1-bit input write port enable
			wren 	=> wren
		);

end architecture test;
