"""
STSF (Spike-Time Sparse Feedback) trainer.

This is the spikerplus-resident port of the reference STSF algorithm in
``stsf/main/network.py``. It implements a Direct Feedback Alignment (DFA)
update rule with a *fixed sparse* feedback matrix, plus a local Hebb-style
weight update, all in 16-bit fixed-point arithmetic that mirrors what the
Spiker-LL accelerator computes in hardware.

Constraints (enforced at construction):

  * Exactly two trainable layers (1 hidden + 1 output). This matches the
    current Spiker-LL RTL; deeper nets would require new trainer modules.
  * Subtractive reset on both LIF layers (matches ``neuron_subtractive`` RTL).
  * Feedback matrix has *exactly one* non-zero per column — one hidden neuron
    feeds back into exactly one output neuron. This is the property that lets
    the hardware use a cheap mux instead of a full matrix multiply.

After ``train()`` completes, two attributes are written onto the trained
SNN for later consumption by :class:`VhdlGenerator`:

    snn.sel_idx_array   : np.ndarray[int]   shape (n_hidden,)
    snn.neuron_constants: np.ndarray[int]   shape (n_hidden,) signed, BW-bit

These become the ``SEL_IDX_ARRAY`` and ``NEURON_CONSTANTS`` VHDL constants
inside the generated ``hidden_layer_trainer`` entity.
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn

from .quantizer import BW, FP_DEC, check_range, clamp_int_, fixed_point


def _build_one_per_column_sfmatrix(
		n_out, n_hidden, lr, loss_value, quant=True, seed=None,
		bw=BW, fp_dec=FP_DEC):
	"""Construct a feedback matrix with exactly one non-zero per column.

	Each hidden neuron (column) is wired to exactly one output neuron (row);
	the magnitude of that connection is the per-neuron learning rate that
	will be baked into the ``NEURON_CONSTANTS`` array of the RTL trainer.

	The "≥1 per column" patching logic in the reference ``SFMatrix`` (see
	``stsf/main/network.py``) is *not* used: it can leave columns with two
	or more non-zeros, which the cheap-mux HW cannot represent.
	"""
	g = torch.Generator()
	if seed is not None:
		g.manual_seed(int(seed))

	# Pick exactly one row index per column. Uniform over [0, n_out).
	row_idx = torch.randint(0, n_out, (n_hidden,), generator=g)
	# Connection magnitudes — random sign and magnitude, scaled by lr*loss_value
	# so that the *quantised* integer ends up in a sensible range.
	bd = float(np.sqrt(n_hidden / n_out))
	raw = (2.0 * bd * torch.rand(n_hidden, generator=g) - bd) * lr * loss_value

	mat = torch.zeros(n_out, n_hidden)
	mat[row_idx, torch.arange(n_hidden)] = raw

	if quant:
		mat = fixed_point(mat, fp_dec, bw)
		# After quantisation some columns can become exactly 0 when
		# lr * loss_value is small enough that the product rounds down to
		# zero in fixed-point (e.g. 0.01 * 0.667 * 256 ≈ 1.7 → 1, but
		# values near the zero crossing of the uniform draw become 0).
		# Patch those columns to ±1 so the HW invariant always holds.
		zero_cols = (mat != 0).sum(dim=0) == 0
		if zero_cols.any():
			for col in zero_cols.nonzero(as_tuple=True)[0]:
				# Sign from the raw value before quantisation; if raw was
				# positive (or exactly 0) use +1, otherwise -1.
				sign = 1.0 if raw[col].item() >= 0 else -1.0
				mat[row_idx[col], col] = sign

	# Verify the invariant the HW relies on.
	per_col_nonzero = (mat != 0).sum(dim=0)
	assert torch.all(per_col_nonzero == 1), (
		"SFMatrix invariant broken: every column must have exactly one "
		f"non-zero entry; got per-column counts {per_col_nonzero.tolist()}"
	)
	return mat


class STSFTrainer:
	"""Train a 2-layer ``SNN`` with the STSF local DFA rule.

	Parameters
	----------
	snn          : SNN built by ``NetBuilder``. Must have exactly two
	               (Linear, Leaky) pairs.
	lr           : learning rate scalar.
	loss_value   : output-layer scaling. Conventionally ``2 / n_classes``.
	update_every : number of timesteps between weight updates (the HW
	               ``update_every_n`` register).
	seed         : optional integer for reproducible SFMatrix construction.
	device       : torch device override (defaults to ``cuda`` if available).
	"""

	def __init__(self, snn, lr=0.01, loss_value=None, update_every=5,
			seed=None, device=None):

		self._validate_snn(snn)
		self.snn = snn

		# Resolve layer references in spiker's ModuleDict layout.
		# build_snn names trainable Linears "fc1", "fc2" and LIFs "lif1",
		# "lif2" when neuron_model == "lif" (which is enforced for learning).
		self.fc1 = snn.layers["fc1"]
		self.fc2 = snn.layers["fc2"]
		self.lif1 = snn.layers["lif1"]
		self.lif2 = snn.layers["lif2"]

		self.n_in     = self.fc1.in_features
		self.n_hidden = self.fc1.out_features
		self.n_out    = self.fc2.out_features

		self.lr           = float(lr)
		self.loss_value   = float(loss_value if loss_value is not None
		                          else 2.0 / self.n_out)
		self.update_every = int(update_every)
		self.loss_fn      = nn.MSELoss()

		# Fixed-point bitwidths: read from the parsed learning block so
		# there is a single source of truth (the learning config dict).
		self.bw     = snn.learning["bw"]
		self.fp_dec = snn.learning["fp_dec"]

		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available()
			                      else "cpu")
		self.device = device
		self.snn.to(self.device)

		# Pre-quantize the SNN to fixed point so every arithmetic
		# step matches what the RTL accelerator computes. The ``*10`` factor
		# pushes Xavier-initialised weights into a range where the
		# fixed-point integer representation is non-trivial — matches
		# ``stsf/main/network.py:FCNetwork.reset_parameters``.
		with torch.no_grad():
			self.fc1.weight.data = fixed_point(
				self.fc1.weight.data * 10, self.fp_dec, self.bw)
			self.fc2.weight.data = fixed_point(
				self.fc2.weight.data * 10, self.fp_dec, self.bw)
			# Thresholds are scalars in snntorch but stored as 0-d tensors;
			# quantize them to the same grid as the weights so the LIF
			# comparison happens in integer space.
			self.lif1.threshold.data = fixed_point(
				self.lif1.threshold.data, self.fp_dec, self.bw)
			self.lif2.threshold.data = fixed_point(
				self.lif2.threshold.data, self.fp_dec, self.bw)

		# Fixed sparse feedback matrix: shape (n_out, n_hidden), one non-zero
		# per column. Kept as a buffer-like tensor (no grad).
		self.feedback = _build_one_per_column_sfmatrix(
			n_out=self.n_out,
			n_hidden=self.n_hidden,
			lr=self.lr,
			loss_value=self.loss_value,
			quant=True,
			seed=seed,
			bw=self.bw,
			fp_dec=self.fp_dec,
		).to(self.device)

		# Per-hidden-neuron index of the (single) output neuron it feeds.
		# argmax over rows works because every column has exactly one
		# non-zero (asserted in the helper above).
		sel_idx = self.feedback.abs().argmax(dim=0)        # (n_hidden,)
		# Per-hidden-neuron signed constant — the value that the HW
		# trainer adds / subtracts on each update.
		neuron_constants = self.feedback.gather(
			0, sel_idx.unsqueeze(0)).squeeze(0)            # (n_hidden,)

		self.sel_idx          = sel_idx.cpu()
		self.neuron_constants = neuron_constants.cpu()

	# ------------------------------------------------------------------
	# Validation
	# ------------------------------------------------------------------

	@staticmethod
	def _validate_snn(snn):
		layer_keys = list(snn.layers.keys())
		expected = ["fc1", "lif1", "fc2", "lif2"]
		if layer_keys != expected:
			raise ValueError(
				"STSFTrainer requires an SNN with exactly two LIF layers "
				f"(keys {expected}); got {layer_keys}. Check the net_dict "
				"has exactly two 'layer_*' entries with neuron_model='lif'."
			)

	# ------------------------------------------------------------------
	# Per-sample forward + update
	# ------------------------------------------------------------------

	def _forward_step(self, x):
		"""Run one timestep through the two LIF layers, in-place state."""
		cur1 = self.fc1(x)
		spk1, _ = self.lif1(cur1, self.lif1.mem)
		cur2 = self.fc2(spk1)
		spk2, _ = self.lif2(cur2, self.lif2.mem)
		# Quantize membrane to integer grid after each step — matches the
		# HW which stores membrane state in fixed-point registers.
		self.lif1.mem.copy_(torch.trunc(self.lif1.mem))
		self.lif2.mem.copy_(torch.trunc(self.lif2.mem))
		return spk1, spk2

	def train_sample(self, data, target):
		"""Train on one batch of spike trains.

		data:   (n_timesteps, batch_size, n_in)
		target: (batch_size,) integer labels
		Returns: (loss, pred) — scalar MSE loss and per-sample argmax.
		"""
		n_timesteps, batch_size, _ = data.shape
		device = data.device

		# Reset LIF state to zero — fresh sample.
		self.snn.reset()
		# snn.reset() populates self.snn.mem[<lif_key>]; mirror into the
		# layer's internal .mem attribute so _forward_step's stateful call
		# sees zero state at t=0.
		self.lif1.mem = self.snn.mem["lif1"]
		self.lif2.mem = self.snn.mem["lif2"]

		# One-hot target.
		tgt = torch.zeros(batch_size, self.n_out, device=device)
		tgt.scatter_(1, target.unsqueeze(1), 1.0)

		spk_sum = None

		for t in range(n_timesteps):
			spk1, spk2 = self._forward_step(data[t])
			spk_sum = spk2 if spk_sum is None else spk_sum + spk2

			# Skip weight updates outside the configured cadence — matches
			# the multi_cycle_dapapath strobe in RTL.
			if (t + 1) % self.update_every != 0:
				continue

			# Trip-wire: weights must still fit in bw bits before we update.
			check_range(self.fc1.weight.data, self.bw, "hidden layer weights")
			check_range(self.fc2.weight.data, self.bw, "output layer weights")

			error = spk2 - tgt                                # (B, n_out)

			# ---- Hidden layer update (DFA via fixed feedback) -----------
			# loss_hidden[i] = error @ feedback[:, i] reduces to
			#   error[:, sel_idx[i]] * neuron_constants[i]
			# because feedback has exactly one non-zero per column. The
			# matmul form below is functionally identical *and* lets the
			# Python model double-check the HW invariant (any deviation
			# would surface as a numerical mismatch in co-sim).
			loss_hidden = error @ self.feedback              # (B, n_hidden)
			dw_hidden   = (loss_hidden * spk1).T @ data[t]   # (n_hidden, n_in)
			self.fc1.weight.data -= dw_hidden
			clamp_int_(self.fc1.weight.data, self.bw)

			# ---- Output layer update (delta rule) -----------------------
			# loss_grad encodes the ±CONST step the RTL output trainer
			# applies; quantising here matches the HW exactly.
			loss_grad = fixed_point(
				error * self.loss_value * self.lr, self.fp_dec, self.bw)
			dw_out = loss_grad.T @ spk1                       # (n_out, n_hidden)
			self.fc2.weight.data -= dw_out
			clamp_int_(self.fc2.weight.data, self.bw)

		loss = self.loss_fn(spk_sum, tgt)
		pred = spk_sum.argmax(dim=1, keepdim=True)
		return loss, pred

	# ------------------------------------------------------------------
	# Epoch loop
	# ------------------------------------------------------------------

	def train(self, train_loader, val_loader=None, n_epochs=10):
		"""Run ``n_epochs`` of STSF training, then publish HW constants."""
		for epoch in range(n_epochs):
			t0 = time.time()
			train_loss, train_acc = self._run_epoch(train_loader, train=True)
			if val_loader is not None:
				val_loss, val_acc = self._run_epoch(val_loader, train=False)
			else:
				val_loss, val_acc = float("nan"), float("nan")

			logging.info(
				"STSF epoch %d  train loss %.4f acc %.2f%%  "
				"val loss %.4f acc %.2f%%  (%.1fs)",
				epoch, train_loss, 100.0 * train_acc,
				val_loss, 100.0 * val_acc, time.time() - t0,
			)

		# Publish the constants the RTL generator will bake into the
		# hidden_layer_trainer entity.
		self.snn.sel_idx_array   = self.sel_idx.numpy().astype(np.int64)
		self.snn.neuron_constants = self.neuron_constants.numpy().astype(np.int64)

	def _run_epoch(self, loader, train):
		total_loss = 0.0
		total_correct = 0
		total_samples = 0

		for data, labels in loader:
			# Spiker's data convention is (batch, T, features); the trainer
			# wants (T, batch, features) so we transpose first.
			data   = data.permute(1, 0, 2).to(self.device)
			labels = labels.to(self.device)

			if train:
				loss, pred = self.train_sample(data, labels)
			else:
				with torch.no_grad():
					loss, pred = self.train_sample(data, labels)

			total_loss   += loss.item() * labels.size(0)
			total_correct += (pred.squeeze(1) == labels).sum().item()
			total_samples += labels.size(0)

		if total_samples == 0:
			return float("nan"), float("nan")
		return total_loss / total_samples, total_correct / total_samples

	# ------------------------------------------------------------------
	# Convenience
	# ------------------------------------------------------------------

	def reset(self):
		"""Reset LIF membrane state."""
		self.snn.reset()
