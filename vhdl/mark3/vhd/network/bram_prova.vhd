	begin
	
		case sel is
			
			when '0' =>
				mux_out <= in0;
		
			when others =>
				mux_out <= in1;

		end case;

	end process selection;


end architecture behaeiour;
