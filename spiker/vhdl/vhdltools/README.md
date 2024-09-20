
# hdltools

## Installation

- Download the repository
- Go to repository directory (we assume it will be placed at '''./hdltools''')
- Inttall with PIP
  - pip install hdltools
  - add '--upgrade' to replace a previous installation.

# VHDL Generator

# Register Bank Generator

## Aim
The purpose of the code in this repository is to auto-generate a bank of registers using VHDL with an interface AXI Lite to be used in programmable logic design.

## Usage
The user must adapt the __main__ part of the regbank_gen.py script to adapt to her/his needs, as will be explained in this section.

### Creating a record bank

The first step is to create and object of the class RegisterBank, e.g.

    myregbank = RegisterBank("myregbank", "rtl", 32, 8)

The class constructor has 4 mandatory parameters:
 1. entity name
 2. architecture name
 3. registers width
 4. number of registers

 ### Adding records
 Once you have a Register Bank, the next step is to populate it with registers. To do so, you must use the RegisterBank.add() method, e.g.

	 myregbank.add(0, "Golden")

where the first parameter is the register number (starting in 0 and up to number of registers - 1) and the register name. Note that all register fields will get as prefix the register name.

 ### Adding fields to records
After creating a registers in your Register Bank, you need to add at least one field to your register. Just call the method Register.add():

    myregbank.reg[0].add("golden","ReadOnly",0,32)

We have 4 parameters:
- field name
- field type (see the supported types below)
- field starting bit (the LSB where it starts)
- field size
	- if the size is 1, the field will have a *std_logic* type
	- otherwise, the field will have *std_logic_vector* type  

### Record Field Types

 The following registers types are supported:

- ReadOnly: the AXI master has read rights, but not write ones. An input port is created to allow the user to control the content of the field.
- ReadWrite: the AXI master has read/write rights. An output port is created to allow the user to read this field, but she/he cannot change its value, unless the variable *externalClear* is set to true. In this case, a input port is included to allow the user to clear the content of this field.
- SplitReadWrite: this field behaves like ReadWrite, but without implementing internally the readback (user logic must take care of it). Hence, not necessarily the master will read what it has written in this field.
- Write2Clear: this field captures a '1' in the user input port. It will only re-trigger when the AXI master clears it.
- Write2Pulse:  if the user writes '1' to this field, it will clear its content and make a one clock cycle pulse. On the AXI side, it is just a read-only register.

Note: the read, write are always from the point of view of the AXI master.

## Generating the VHDL file

To auto-generate the .vhd file, the user must call the routine RegisterBank.write_file(). The created file will be saved in the *output* folder.
