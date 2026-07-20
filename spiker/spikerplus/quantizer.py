"""
Fixed-point quantization helpers used by the STSF on-chip learning rule.

These functions mirror the arithmetic of the Spiker-LL hardware (16-bit signed,
8 fractional bits). They are ported from ``stsf/main/quantizer.py`` so that the
``spikerplus`` package is self-contained and does not pull a cross-tree
dependency on the reference STSF implementation.

Module-level constants ``FP_DEC`` and ``BW`` define the canonical hardware
fixed-point format. Any user-supplied ``bitwidth_config`` must agree with them
when learning is enabled (see ``VhdlGenerator``).
"""

import numpy as np
import torch


# Canonical hardware fixed-point format.
FP_DEC = 8   # fractional bits
BW     = 16  # total signed bit-width


def fixed_point(value, fp_dec=FP_DEC, bitwidth=BW):
	"""Quantize ``value`` to ``Q(bitwidth-fp_dec).fp_dec`` fixed-point.

	The value is scaled by 2**fp_dec to push the desired fractional bits into
	the integer part, then truncated and saturated to the signed range.
	"""
	quant = value * 2**fp_dec
	return saturated_int(quant, bitwidth)


def saturated_int(value, bitwidth):
	"""Truncate to integer then saturate to the signed ``bitwidth`` range."""
	return saturate(to_int(value), bitwidth)


def saturate(value, bitwidth):
	"""Clip ``value`` into the signed range ``[-2**(bw-1), 2**(bw-1)-1]``.

	Works on Python scalars, numpy arrays, and torch tensors. Out-of-range
	values are clipped in place for arrays/tensors (a debug message is also
	printed, matching the reference behaviour).
	"""
	qmax =  2**(bitwidth-1) - 1
	qmin = -2**(bitwidth-1)

	if type(value).__module__ == np.__name__ or \
			type(value).__module__ == torch.__name__:
		# Diagnose out-of-range elements before clipping (helps catch a
		# misconfigured learning rate that would silently saturate weights).
		if type(value).__module__ == np.__name__:
			oor = np.where((value > qmax) | (value < qmin))
			if oor[0].size > 0:
				print(f"Values out of range in numpy array: {value[oor]}")
		else:
			oor = torch.where((value > qmax) | (value < qmin))
			if oor[0].numel() > 0:
				print(f"Values out of range in torch tensor: {value[oor]}")
				print(f"Valid range: [{qmin}, {qmax}]")
		value[value > qmax] = qmax
		value[value < qmin] = qmin
		return value.float()

	# Plain Python number
	if value > qmax:
		value = qmax
	elif value < qmin:
		value = qmin
	return float(value)


def to_int(value):
	"""Truncate towards zero, preserving the input container type."""
	if type(value).__module__ == np.__name__:
		return value.astype(int).astype(float)
	if type(value).__module__ == torch.__name__:
		return value.type(torch.int64).float()
	return float(int(value))


def check_range(tensor, bitwidth, name=""):
	"""Assert that a tensor still fits inside the signed ``bitwidth`` range.

	Used after a weight update as a tripwire: a failure here means the update
	step has driven weights outside the representable HW range, which would
	silently overflow in the accelerator.
	"""
	qmin = -2**(bitwidth-1)
	qmax =  2**(bitwidth-1) - 1
	mn = tensor.min().item()
	mx = tensor.max().item()
	assert mn >= qmin and mx <= qmax, (
		f"{name} out of range: [{mn}, {mx}] should be inside "
		f"[{qmin}, {qmax}]"
	)


def clamp_int_(t: torch.Tensor, bitwidth: int = BW):
	"""In-place: floor to integer and clip to signed ``bitwidth`` range."""
	qmin = -(1 << (bitwidth - 1))
	qmax =  (1 << (bitwidth - 1)) - 1
	t.floor_()
	t.clamp_(qmin, qmax)
	return t
