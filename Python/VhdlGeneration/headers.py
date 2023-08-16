bram_header = """
--BRAM_SINGLE_MACRO : In order to incorporate this function into the design,
--     VHDL      : the following instance declaration needs to be placed
--   instance    : in the architecture body of the design code.  The
--  declaration  : (BRAM_SINGLE_MACRO_inst) and/or the port declarations
--     code      : after the "=>" assignment maybe changed to properly
--               : reference and connect this function to the design.
--               : All inputs and outputs must be connected.

--    Library    : In addition to adding the instance declaration, a use
--  declaration  : statement for the UNISIM.vcomponents library needs to be
--      for      : added before the entity declaration.  This library
--    Xilinx     : contains the component declarations for all Xilinx
--   primitives  : primitives and points to the models that will be used
--               : for simulation.

--  Copy the following four statements and paste them before the
--  Entity declaration, unless they already exist.
"""

bram_description = """
-- BRAM_SINGLE_MACRO: Single Port RAM
--                    Artix-7
-- Xilinx HDL Language Template, version 2022.2

-- Note -  This Unimacro model assumes the port directions to be "downto". 
--         Simulation of this model with "to" in the port directions could 
--	   lead to erroneous results.

---------------------------------------------------------------------
--  READ_WIDTH | BRAM_SIZE | READ Depth  | ADDR Width |            --
-- WRITE_WIDTH |           | WRITE Depth |            |  WE Width  --
-- ============|===========|=============|============|============--
--    37-72    |  "36Kb"   |      512    |    9-bit   |    8-bit   --
--    19-36    |  "36Kb"   |     1024    |   10-bit   |    4-bit   --
--    19-36    |  "18Kb"   |      512    |    9-bit   |    4-bit   --
--    10-18    |  "36Kb"   |     2048    |   11-bit   |    2-bit   --
--    10-18    |  "18Kb"   |     1024    |   10-bit   |    2-bit   --
--     5-9     |  "36Kb"   |     4096    |   12-bit   |    1-bit   --
--     5-9     |  "18Kb"   |     2048    |   11-bit   |    1-bit   --
--     3-4     |  "36Kb"   |     8192    |   13-bit   |    1-bit   --
--     3-4     |  "18Kb"   |     4096    |   12-bit   |    1-bit   --
--       2     |  "36Kb"   |    16384    |   14-bit   |    1-bit   --
--       2     |  "18Kb"   |     8192    |   13-bit   |    1-bit   --
--       1     |  "36Kb"   |    32768    |   15-bit   |    1-bit   --
--       1     |  "18Kb"   |    16384    |   14-bit   |    1-bit   --
---------------------------------------------------------------------

"""
