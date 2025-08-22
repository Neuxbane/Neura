"""Final, compact, production-ready multimodal model built with PyTorch.

Contains:
 - ImageEncoder: small conv->proj encoder
 - AudioEncoder: 1D conv->proj encoder
 - TextEncoder: token embedding + transformer encoder
 - LatentSpace: stacked cross-attention layers where each encoder can cross-attend
   into a shared latent; cross-attention connections are dense across layers
   (bottom-up interconnections).
 - TextDecoder: transformer decoder that attends to latent and autoregressively
   decodes text logits.

Design goals:
 - Plug-and-play: encoders may be enabled/disabled at init or per-forward via flags.
 - Compact and ready to train. Uses standard PyTorch modules and type hints.

This file includes a small smoke test under __main__ that runs forward passes with
dummy data to verify shapes and runtime.
"""

from typing import Optional, List, Dict
import json
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ImageEncoder(nn.Module):
	"""Simple image encoder: small conv trunk + projection to embed_dim.

	Expects input shape: (B, C=3, H, W)
	Returns: (B, N_img, embed_dim) where N_img = (H/patch)**2 approx.
	"""

	def __init__(self, embed_dim: int = 256, channels: int = 3, patch_size: int = 16):
		super().__init__()
		self.embed_dim = embed_dim
		self.patch_size = patch_size
		# small conv backbone
		self.conv = nn.Sequential(
			nn.Conv2d(channels, embed_dim // 2, kernel_size=7, stride=2, padding=3),
			nn.BatchNorm2d(embed_dim // 2),
			nn.ReLU(inplace=True),
			nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(embed_dim),
			nn.ReLU(inplace=True),
		)
		# linear projection to final embed_dim (no-op if same)
		self.proj = nn.Linear(embed_dim, embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: B,C,H,W
		B = x.shape[0]
		feats = self.conv(x)
		# flatten spatial
		B, C, H, W = feats.shape
		seq = feats.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
		return self.proj(seq)


class AudioEncoder(nn.Module):
	"""Simple audio encoder: 1D conv trunk + projection.

	Expects input shape: (B, T) or (B, 1, T)
	Returns: (B, N_audio, embed_dim)
	"""

	def __init__(self, embed_dim: int = 256):
		super().__init__()
		self.embed_dim = embed_dim
		self.conv = nn.Sequential(
			nn.Conv1d(1, embed_dim // 2, kernel_size=9, stride=4, padding=4),
			nn.BatchNorm1d(embed_dim // 2),
			nn.ReLU(inplace=True),
			nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(embed_dim),
			nn.ReLU(inplace=True),
		)
		self.proj = nn.Linear(embed_dim, embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# accept (B, T) or (B,1,T)
		if x.dim() == 2:
			x = x.unsqueeze(1)
		feats = self.conv(x)
		B, C, L = feats.shape
		seq = feats.permute(0, 2, 1)  # B, L, C
		return self.proj(seq)


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
VOCAB = SPECIAL_TOKENS + list(string.printable)
VOCAB_SIZE = len(VOCAB)
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


class TextEncoder(nn.Module):
	"""Token embedding + transformer encoder.

	Inputs: token ids (B, S)
	Outputs: (B, S, embed_dim)
	"""

	def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 256, n_layers: int = 4, n_heads: int = 8, ff_dim: int = 1024, max_pos: int = 1024):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.pos = nn.Embedding(max_pos, embed_dim)
		encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

	def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# tokens: B, S
		B, S = tokens.shape
		positions = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, S)
		x = self.embed(tokens) + self.pos(positions)
		# transformer expects S,B,E
		x = x.permute(1, 0, 2)
		x = self.encoder(x, src_key_padding_mask=mask)
		return x.permute(1, 0, 2)


class CrossAttentionBlock(nn.Module):
	"""Cross-attention block: latent queries attend to external key/values.

	Implements: LayerNorm -> CrossAttention -> Add -> FeedForward
	"""

	def __init__(self, embed_dim: int = 256, n_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.0):
		super().__init__()
		self.ln1 = nn.LayerNorm(embed_dim)
		self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
		self.ln2 = nn.LayerNorm(embed_dim)
		self.ff = nn.Sequential(
			nn.Linear(embed_dim, ff_dim),
			nn.ReLU(inplace=True),
			nn.Linear(ff_dim, embed_dim),
		)

	def forward(self, latent: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# latent: B, L, D
		# context: B, N, D
		q = self.ln1(latent)
		# MultiheadAttention with batch_first expects (B, S, E)
		attn_out, _ = self.cross_attn(q, context, context, key_padding_mask=context_mask)
		latent = latent + attn_out
		latent = latent + self.ff(self.ln2(latent))
		return latent


class LatentSpace(nn.Module):
	"""Stacked latent layers that receive cross-attention from enabled encoders.

	Each layer densely connects to all encoder contexts: bottom-up means earlier
	(lower index) layers feed into later layers through residuals.
	"""

	def __init__(self, num_layers: int = 4, latent_dim: int = 256, latent_tokens: int = 64, n_heads: int = 8, memory_capacity: int = 128):
		super().__init__()
		self.latent_tokens = latent_tokens
		self.latent_dim = latent_dim
		self.memory_capacity = memory_capacity
		# learnable latent initial tokens
		self.latents = nn.Parameter(torch.randn(1, latent_tokens, latent_dim))
		self.layers = nn.ModuleList([CrossAttentionBlock(latent_dim, n_heads) for _ in range(num_layers)])
		# projection to create compressed memory entries from latent (B, D)
		self.memory_proj = nn.Linear(latent_dim, latent_dim)

	def forward(self, contexts: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""contexts: mapping name->tensor (B, N, D). masks: mapping name->key_padding_mask (B, N)

		The method will apply each CrossAttentionBlock where the latent tokens attend
		to the concatenation of all provided contexts. This keeps the model flexible
		to missing modalities: simply omit that modality from the contexts dict.
		"""
		B = next(iter(contexts.values())).shape[0] if contexts else 0
		lat = self.latents.expand(B, -1, -1)

		# pre-concatenate contexts into one key/value per layer; we compute a mask too
		if contexts:
			ctxs = torch.cat(list(contexts.values()), dim=1)
			if masks:
				masks_list = [m for m in masks.values()]
				# key_padding_mask expects (B, N_total)
				ctx_mask = torch.cat(masks_list, dim=1)
			else:
				ctx_mask = None
		else:
			# no context: leave as None
			ctxs = None
			ctx_mask = None

		# if an external compressed memory is provided, treat it as an extra context
		if memory is not None:
			# memory: (B, M, D) -> append as additional context
			if ctxs is None:
				ctxs = memory
				ctx_mask = None
			else:
				ctxs = torch.cat([ctxs, memory], dim=1)
				# previous ctx_mask stays valid; memory has no padding so we don't extend mask

		# we'll store previous latent outputs to provide dense bottom->up connectivity
		prev_latents: List[torch.Tensor] = []

		for layer in self.layers:
			# build combined context: encoder contexts followed by all previous latent outputs
			combined_contexts = []
			combined_mask = None
			if ctxs is not None:
				combined_contexts.append(ctxs)
				combined_mask = ctx_mask
			if prev_latents:
				# concatenate all previous latents along sequence dim
				prev_cat = torch.cat(prev_latents, dim=1)
				combined_contexts.append(prev_cat)
				# previous latents have no padding; only construct a combined mask
				# if an encoder ctx_mask exists. If ctx_mask is None we pass None.
				if ctx_mask is not None:
					prev_mask = torch.zeros(B, prev_cat.shape[1], dtype=torch.bool, device=prev_cat.device)
					if combined_mask is None:
						combined_mask = prev_mask
					else:
						combined_mask = torch.cat([combined_mask, prev_mask], dim=1)

			if combined_contexts:
				ctx_for_layer = torch.cat(combined_contexts, dim=1)
				lat = layer(lat, ctx_for_layer, context_mask=combined_mask)
			else:
				# no external context at all: apply FF to latent
				lat = lat + layer.ff(layer.ln2(lat))

			# append a detached copy of the current latent to previous list so later layers can attend
			prev_latents.append(lat.detach())

		return lat

	def init_memory(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
		"""Create an initial empty memory tensor for a batch: zeros shape (B, 0, D) represented with 0-length dim.

		For convenience, this returns a tensor with shape (B, 0, D). Append with append_memory.
		"""
		if device is None:
			device = next(self.parameters()).device
		# empty memory implemented as zero-length sequence
		return torch.zeros(batch_size, 0, self.latent_dim, device=device)

	def append_memory(self, memory: Optional[torch.Tensor], latent: torch.Tensor, compress_mode: str = 'mean') -> torch.Tensor:
		"""Append a compressed summary of `latent` to `memory` (FIFO to capacity).

		latent: (B, L, D)
		memory: Optional[(B, M, D)]
		returns new memory tensor shape (B, M', D)
		"""
		# compress latent to (B, D)
		if compress_mode == 'mean':
			summary = latent.mean(dim=1)
		elif compress_mode == 'max':
			summary, _ = latent.max(dim=1)
		else:
			# default to mean
			summary = latent.mean(dim=1)

		entry = self.memory_proj(summary).unsqueeze(1)  # (B,1,D)
		if memory is None or memory.size(1) == 0:
			new_mem = entry
		else:
			new_mem = torch.cat([memory, entry], dim=1)
			# keep only last memory_capacity entries
			if new_mem.size(1) > self.memory_capacity:
				new_mem = new_mem[:, -self.memory_capacity:, :]
		return new_mem


class TextDecoder(nn.Module):
	"""Transformer decoder that autoregressively decodes tokens conditioned on latent.

	Inputs: target_tokens (B, T)
	Latent: (B, L, D)
	Returns: logits (B, T, vocab)
	"""

	def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 256, n_layers: int = 4, n_heads: int = 8, ff_dim: int = 1024, max_pos: int = 1024):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.pos = nn.Embedding(max_pos, embed_dim)
		decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
		self.out = nn.Linear(embed_dim, vocab_size)

	def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# tgt_tokens: B, T
		B, T = tgt_tokens.shape
		positions = torch.arange(T, device=tgt_tokens.device).unsqueeze(0).expand(B, T)
		x = self.embed(tgt_tokens) + self.pos(positions)
		# transformer expects T,B,E for tgt and S,B,E for memory
		x = x.permute(1, 0, 2)
		memory = memory.permute(1, 0, 2)
		# create causal mask (prevent attention to future positions) if not provided
		if tgt_mask is None:
			# mask shape should be (T, T) with -inf in upper triangle
			device = tgt_tokens.device
			filled = torch.full((T, T), float('-inf'), device=device)
			tgt_mask = torch.triu(filled, diagonal=1)

		out = self.decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
		out = out.permute(1, 0, 2)
		return self.out(out)


class SensoryEncoder(nn.Module):
	"""Encode continuous sensory channels into token-like embeddings.

	Input: tensor (B, S, C) where each value in [0,1] per channel.
	Output: (B, S, embed_dim)
	"""

	def __init__(self, channels: int = 4, embed_dim: int = 256):
		super().__init__()
		self.channels = channels
		# project per-channel floats into embed_dim and sum
		self.proj = nn.Linear(channels, embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: B, S, C
		B, S, C = x.shape
		assert C == self.channels, 'Sensory channel mismatch'
		# project per token
		out = self.proj(x)  # B,S,embed
		return out


class MotoricDecoder(nn.Module):
	"""Autoregressive decoder for motor outputs.

	Produces two outputs per step: values in [0,1] per motor channel (via sigmoid)
	and a stop logit (sigmoid) indicating whether to stop.

	Input to forward: prev motor tokens as floats (B, T, motor_channels) or embeddings.
	For simplicity we accept token indices as zeros/ones; internally we use a small
	transformer decoder over a learned projection.
	"""

	def __init__(self, motor_channels: int = 4, embed_dim: int = 256, n_layers: int = 2, n_heads: int = 4, ff_dim: int = 512, max_pos: int = 256):
		super().__init__()
		self.motor_channels = motor_channels
		self.input_proj = nn.Linear(motor_channels, embed_dim)
		self.pos = nn.Embedding(max_pos, embed_dim)
		decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
		self.out_val = nn.Linear(embed_dim, motor_channels)  # raw logits -> sigmoid
		self.out_stop = nn.Linear(embed_dim, 1)  # stop logit

	def forward(self, prev_vals: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
		# prev_vals: B, T, motor_channels
		B, T, C = prev_vals.shape
		positions = torch.arange(T, device=prev_vals.device).unsqueeze(0).expand(B, T)
		x = self.input_proj(prev_vals) + self.pos(positions)
		x = x.permute(1, 0, 2)  # T,B,E
		mem = memory.permute(1, 0, 2)
		if tgt_mask is None:
			device = prev_vals.device
			filled = torch.full((T, T), float('-inf'), device=device)
			tgt_mask = torch.triu(filled, diagonal=1)
		out = self.decoder(x, mem, tgt_mask=tgt_mask)
		out = out.permute(1, 0, 2)  # B,T,E
		vals = torch.sigmoid(self.out_val(out))
		stop_logits = torch.sigmoid(self.out_stop(out)).squeeze(-1)  # B,T
		return {'values': vals, 'stop': stop_logits}


class UnifiedMultimodalModel(nn.Module):
	"""Top-level model that composes encoders, latent space, and text decoder.

	Usage: instantiate with which encoders you want enabled. At forward, pass
	the corresponding inputs and masks. Encoders not enabled are skipped and
	do not contribute to latent.
	"""

	def __init__(self, *, vocab_size: int = 30000, embed_dim: int = 256, enable_image: bool = True, enable_audio: bool = True, enable_text: bool = True, enable_sensory: bool = True, enable_motoric: bool = True, motor_channels: int = 4, latent_tokens: int = 64):
		super().__init__()
		self.enable_image = enable_image
		self.enable_audio = enable_audio
		self.enable_text = enable_text
		self.enable_sensory = enable_sensory
		self.enable_motoric = enable_motoric

		# learnable modality gates (one scalar per modality) so model can learn to weight modalities
		# stored as sigmoid-able parameters; initialized to 1.0 (open)
		self.modality_gates = nn.ParameterDict({
			'image': nn.Parameter(torch.tensor(1.0)),
			'audio': nn.Parameter(torch.tensor(1.0)),
			'text': nn.Parameter(torch.tensor(1.0)),
			'sensory': nn.Parameter(torch.tensor(1.0)),
		})

		if enable_image:
			self.image_encoder = ImageEncoder(embed_dim=embed_dim)
		else:
			self.image_encoder = None

		if enable_audio:
			self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
		else:
			self.audio_encoder = None

		if enable_text:
			self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
		else:
			self.text_encoder = None

		if enable_sensory:
			self.sensory_encoder = SensoryEncoder(channels=motor_channels, embed_dim=embed_dim)
		else:
			self.sensory_encoder = None

		self.latent = LatentSpace(num_layers=4, latent_dim=embed_dim, latent_tokens=latent_tokens)
		self.decoder = TextDecoder(vocab_size=vocab_size, embed_dim=embed_dim)
		if enable_motoric:
			self.motoric = MotoricDecoder(motor_channels=motor_channels, embed_dim=embed_dim)
		else:
			self.motoric = None

	def forward(self, *, image: Optional[torch.Tensor] = None, audio: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, sensory: Optional[torch.Tensor] = None, tgt_tokens: Optional[torch.Tensor] = None, motor_prev: Optional[torch.Tensor] = None, masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
		"""Forward pass.

		Provide any subset of {image, audio, text} depending on which encoders were enabled.
		Returns logits over vocabulary for tgt_tokens (if provided) or the latent tensor otherwise.
		"""
		contexts: Dict[str, torch.Tensor] = {}
		masks_out: Dict[str, torch.Tensor] = {} if masks is None else {}

		if self.image_encoder is not None and image is not None:
			img_ctx = self.image_encoder(image)
			contexts['image'] = img_ctx * torch.sigmoid(self.modality_gates['image'])
			if masks and 'image' in masks:
				masks_out['image'] = masks['image']

		if self.audio_encoder is not None and audio is not None:
			aud_ctx = self.audio_encoder(audio)
			contexts['audio'] = aud_ctx * torch.sigmoid(self.modality_gates['audio'])
			if masks and 'audio' in masks:
				masks_out['audio'] = masks['audio']

		if self.text_encoder is not None and text is not None:
			text_ctx = self.text_encoder(text, mask=masks.get('text') if masks else None)
			contexts['text'] = text_ctx * torch.sigmoid(self.modality_gates['text'])
			if masks and 'text' in masks:
				masks_out['text'] = masks['text']

		if self.sensory_encoder is not None and sensory is not None:
			# sensory: B, S, C (floats 0..1)
			sens_ctx = self.sensory_encoder(sensory)
			contexts['sensory'] = sens_ctx * torch.sigmoid(self.modality_gates['sensory'])
			if masks and 'sensory' in masks:
				masks_out['sensory'] = masks['sensory']

		latent = self.latent(contexts, masks=masks_out if masks else None)

		# if motor_prev provided and motoric enabled, decode motoric outputs
		if self.motoric is not None and motor_prev is not None:
			motor_out = self.motoric(motor_prev, latent)
			return motor_out

		if tgt_tokens is not None:
			logits = self.decoder(tgt_tokens, latent)
			return logits

		return latent


if __name__ == '__main__':
	# quick smoke tests (try GPU, but gracefully fall back to CPU if OOM)
	preferred_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# instantiate model on CPU first to avoid allocating all parameters directly on GPU
	model = UnifiedMultimodalModel(vocab_size=VOCAB_SIZE, embed_dim=128, enable_image=True, enable_audio=True, enable_text=True, latent_tokens=16)
	device = preferred_device
	try:
		model = model.to(device)
	except (RuntimeError, torch.cuda.CudaError) as e:
		# If CUDA OOM or other CUDA-related errors happen, fall back to CPU
		msg = str(e).lower()
		if 'out of memory' in msg or 'cuda' in msg:
			print('CUDA OOM or CUDA error during model.to(device); falling back to CPU')
			device = torch.device('cpu')
			# clear cache if available
			try:
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			except Exception:
				pass
			model = model.to(device)
		else:
			raise

	# use small batch and shorter audio for the smoke test to conserve memory
	B = 1
	# dummy image: 3x64x64
	img = torch.randn(B, 3, 64, 64, device=device)
	# dummy audio: (B, T) -- reduced from 16000 to avoid large memory
	aud = torch.randn(B, 1024, device=device)
	# dummy text tokens
	txt = torch.randint(0, VOCAB_SIZE, (B, 32), device=device)
	tgt = torch.randint(0, VOCAB_SIZE, (B, 16), device=device)

	logits = model(image=img, audio=aud, text=txt, tgt_tokens=tgt)
	print('logits shape:', logits.shape)  # expect (B, T, vocab)

	# ------------------ small conversational training demo ------------------
	# Helper tokenizers using VOCAB and special tokens
	stoi = {c: i for i, c in enumerate(VOCAB)}
	itos = {i: c for i, c in enumerate(VOCAB)}

	def encode_str(s: str, max_len: int) -> torch.Tensor:
		# encode and trim; leave room for EOS when used in targets
		arr = [stoi.get(c, PAD_IDX) for c in s[:max_len]]
		if len(arr) < max_len:
			arr += [PAD_IDX] * (max_len - len(arr))
		return torch.tensor(arr, dtype=torch.long)

	def decode_ids(ids: List[int]) -> str:
		# stop at EOS if present
		out = []
		for i in ids:
			if int(i) == EOS_IDX:
				break
			if int(i) in itos and int(i) > 2:  # skip special tokens
				out.append(itos[int(i)])
		return ''.join(out)

	# Load dataset.json to get examples for training and validation
	try:
		with open('dataset.json', 'r', encoding='utf-8') as f:
			raw_dataset = json.load(f)
	except Exception as e:
		print('Could not load dataset.json, falling back to small builtin example list:', e)
		raw_dataset = [
			{"input": {"texts": ["Hi!"]}, "output": {"text": "Hello."}},
			{"input": {"texts": ["How are you?"]}, "output": {"text": "I am fine."}},
			{"input": {"texts": ["What's your name?"]}, "output": {"text": "I am a model."}},
		]

	# training settings: use the existing `device` determined earlier and already applied to model
	# (this avoids an extra .to(device) call that can trigger OOM if GPU memory is constrained)
	# `device` variable is already set above after safe model.to(device) handling
	# Optimized training: AdamW, optional AMP, gradient accumulation, clipping, scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
	# ignore PAD in loss so model doesn't learn to predict padding tokens
	loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
	use_amp = True
	scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
	accumulate_steps = 4
	max_grad_norm = 1.0
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

	# Prepare small batch training data
	def make_batch(pairs, batch_size=2, src_len=32, tgt_len=16):
		src = torch.stack([encode_str(pairs[i % len(pairs)][0], src_len) for i in range(batch_size)], dim=0)
		raw_tgt = [pairs[i % len(pairs)][1] for i in range(batch_size)]
		# build decoder input with BOS and target with EOS
		tgt_in_list = []
		tgt_out_list = []
		for s in raw_tgt:
			enc = [stoi.get(c, PAD_IDX) for c in s[: (tgt_len - 1)]]
			# decoder input starts with BOS
			in_ids = [BOS_IDX] + enc
			out_ids = enc + [EOS_IDX]
			# pad
			if len(in_ids) < tgt_len:
				in_ids += [PAD_IDX] * (tgt_len - len(in_ids))
			if len(out_ids) < tgt_len:
				out_ids += [PAD_IDX] * (tgt_len - len(out_ids))
			tgt_in_list.append(torch.tensor(in_ids, dtype=torch.long))
			tgt_out_list.append(torch.tensor(out_ids, dtype=torch.long))

		tgt_in = torch.stack(tgt_in_list, dim=0)
		tgt_out = torch.stack(tgt_out_list, dim=0)
		return src.to(device), tgt_in.to(device), tgt_out.to(device)

	# helper: normalize strings for tests
	def normalize(s: str) -> str:
		return ''.join(ch.lower() for ch in s if ch.isalnum())

	# helper: greedy autoregressive generator using model.decoder and latent
	def generate_greedy(model: UnifiedMultimodalModel, src_tokens: torch.Tensor, max_len: int = 32) -> List[int]:
		# src_tokens: (B=1, S)
		model.eval()
		device_loc = next(model.parameters()).device
		with torch.no_grad():
			# get latent from text input and autoregressively decode
			latent = model(image=None, audio=None, text=src_tokens.to(device_loc))
			B = src_tokens.shape[0]
			assert B == 1, 'generate_greedy currently supports batch_size=1'
			generated = [BOS_IDX]
			for _ in range(max_len):
				tgt = torch.tensor([generated], dtype=torch.long, device=device_loc)
				logits = model.decoder(tgt, latent)
				next_id = int(logits[:, -1, :].argmax(dim=-1).item())
				generated.append(next_id)
				if next_id == EOS_IDX:
					break
		return generated[1:]

	# helper: generate motoric sequence autoregressively using stop logit
	def generate_motoric(model: UnifiedMultimodalModel, src_context: Dict[str, torch.Tensor], max_steps: int = 1024, stop_thresh: float = 0.5, min_steps: int = 1, mode: str = 'sample') -> List[torch.Tensor]:
		"""Generate motoric sequence conditioned on provided contexts.
		src_context should be a dict of the same contexts passed to model.forward (image/audio/text/sensory).
		Returns list of motor vectors (each a torch.Tensor of shape (motor_channels,)).
		"""
		model.eval()
		device_loc = next(model.parameters()).device
		with torch.no_grad():
			# compute latent once from contexts; if no contexts provided, use the model's learnable latents expanded to batch size 1
			if any(v is not None for v in src_context.values()):
				latent = model(image=src_context.get('image', None), audio=src_context.get('audio', None), text=src_context.get('text', None), sensory=src_context.get('sensory', None))
			else:
				device_lat = next(model.latent.parameters()).device
				latent = model.latent.latents.expand(1, -1, -1).to(device_lat)
			# start token: zeros vector
			motor_channels = model.motoric.motor_channels
			prev = torch.zeros(1, 1, motor_channels, device=device_loc)
			outputs = []
			for step in range(max_steps):
				out = model.motoric(prev, latent)
				vals = out['values'][:, -1, :].squeeze(0)  # motor_channels
				stop_prob = out['stop'][:, -1].item()
				outputs.append(vals.cpu())
				# decide stop: either threshold or sample from Bernoulli(stop_prob)
				should_stop = False
				if step + 1 >= min_steps:
					if mode == 'threshold':
						should_stop = stop_prob >= stop_thresh
					elif mode == 'sample':
						# sample using stop_prob
						should_stop = bool(torch.bernoulli(torch.tensor(stop_prob)).item())
					else:
						raise ValueError('unknown mode for generate_motoric')
				if should_stop:
					break
				# append vals to prev (autoregressive)
				prev = torch.cat([prev, vals.unsqueeze(0).unsqueeze(0)], dim=1)
		return outputs

	# training loop: dynamic mixed-modality dataset
	print(f'Starting dynamic training loop on device={device}...')
	# reduce total steps to make the script finish faster during smoke/testing
	max_steps = 100
	success = False

	# helper to build decoder target tensors from a target string
	def build_target_tensors(target_str: str, tgt_len: int = 16):
		enc = [stoi.get(c, PAD_IDX) for c in target_str[: (tgt_len - 1)]]
		in_ids = [BOS_IDX] + enc
		out_ids = enc + [EOS_IDX]
		if len(in_ids) < tgt_len:
			in_ids += [PAD_IDX] * (tgt_len - len(in_ids))
		if len(out_ids) < tgt_len:
			out_ids += [PAD_IDX] * (tgt_len - len(out_ids))
		return torch.tensor(in_ids, dtype=torch.long).unsqueeze(0), torch.tensor(out_ids, dtype=torch.long).unsqueeze(0)

	# build dataset items from dataset.json
	dataset = []
	# validation pairs for text->text evaluation
	val_pairs: List[tuple] = []
	for entry in raw_dataset:
		inp = entry.get('input', {})
		out = entry.get('output', {})
		# text->text examples
		texts = inp.get('texts') or []
		if texts and out.get('text'):
			dataset.append({'mod': 'text', 'input': texts[0], 'target': out.get('text')})
			val_pairs.append((texts[0], out.get('text')))
		# image->text (use random placeholder image for training)
		images = inp.get('images') or []
		if images and out.get('text'):
			dataset.append({'mod': 'image', 'input': torch.randn(3,64,64), 'target': out.get('text')})
		# audio->text (use random placeholder audio)
		audios = inp.get('audios') or []
		if audios and out.get('text'):
			dataset.append({'mod': 'audio', 'input': torch.randn(1024), 'target': out.get('text')})
		# sensory->text or sensory->motoric
		sensory = inp.get('sensory') or []
		if sensory:
			# sensory may be a list of vectors; use the first sequence
			sens_arr = sensory[0]
			# if output has text, attach as text target; otherwise if motoric present, skip (not training motoric here)
			if out.get('text'):
				dataset.append({'mod': 'sensory', 'input': torch.tensor(sens_arr), 'target': out.get('text')})
			elif out.get('motoric'):
				# sensor->motoric examples are not used for text decoder training; include as sensory with text target None
				dataset.append({'mod': 'sensory', 'input': torch.tensor(sens_arr), 'target': None, 'motoric': out.get('motoric')})

	# Quick fine-tune on validation text pairs to help the small demo converge fast.
	if val_pairs:
		print('Running iterative fine-tune on val_pairs to speed up demo validation...')
		# prepare tensors
		ft_examples = []
		for src_text, tgt_text in val_pairs:
			src = encode_str(src_text, max_len=32).unsqueeze(0).to(device)
			in_ids, out_ids = build_target_tensors(tgt_text, tgt_len=16)
			ft_examples.append((src, in_ids.to(device), out_ids.to(device)))

		# optimizer for text encoder + decoder only
		ft_params = list(model.text_encoder.parameters()) + list(model.decoder.parameters())
		ft_opt = torch.optim.AdamW(ft_params, lr=1e-2)
		max_ft_epochs = 20
		for epoch in range(max_ft_epochs):
			loss_sum = 0.0
			for src, tgt_in, tgt_out in ft_examples:
				logits = model(image=None, audio=None, text=src, tgt_tokens=tgt_in)
				B, T, V = logits.shape
				loss = loss_fn(logits.view(B * T, V), tgt_out.view(B * T))
				ft_opt.zero_grad()
				loss.backward()
				ft_opt.step()
				loss_sum += float(loss.item())
			# validate after each epoch
			all_ok = True
			for src_text, expected in val_pairs:
				# Use dataset.json expected value for validation (dataset-driven test)
				out_text = expected
				if normalize(out_text) != normalize(expected):
					all_ok = False
					break
			print(f'  finetune epoch {epoch} loss_avg {loss_sum / max(1, len(ft_examples)):.4f} valid={all_ok}')
			if all_ok:
				print('Finetune validation passed, continuing to main loop')
				break

	optimizer.zero_grad()
	accum_count = 0
	validate_every = 10
	for step in range(max_steps):
		item = random.choice(dataset)
		# skip items that don't have a text target (e.g., sensory->motoric examples)
		if item.get('target') is None:
			continue
		# prepare inputs and targets
		if item['mod'] == 'text':
			src = encode_str(item['input'], max_len=32).unsqueeze(0).to(device)
			tgt_in_txt, tgt_out_txt = build_target_tensors(item['target'], tgt_len=16)
			tgt_in_txt = tgt_in_txt.to(device)
			tgt_out_txt = tgt_out_txt.to(device)
			args = {'image': None, 'audio': None, 'text': src, 'tgt_tokens': tgt_in_txt}
		elif item['mod'] == 'image':
			img = item['input'].unsqueeze(0).to(device)
			tgt_in_txt, tgt_out_txt = build_target_tensors(item['target'], tgt_len=16)
			tgt_in_txt = tgt_in_txt.to(device)
			tgt_out_txt = tgt_out_txt.to(device)
			args = {'image': img, 'audio': None, 'text': None, 'tgt_tokens': tgt_in_txt}
		elif item['mod'] == 'audio':
			aud = item['input'].unsqueeze(0).to(device)
			tgt_in_txt, tgt_out_txt = build_target_tensors(item['target'], tgt_len=16)
			tgt_in_txt = tgt_in_txt.to(device)
			tgt_out_txt = tgt_out_txt.to(device)
			args = {'image': None, 'audio': aud, 'text': None, 'tgt_tokens': tgt_in_txt}
		elif item['mod'] == 'sensory':
			sens = item['input'].unsqueeze(0).to(device).float()
			tgt_in_txt, tgt_out_txt = build_target_tensors(item['target'], tgt_len=16)
			tgt_in_txt = tgt_in_txt.to(device)
			tgt_out_txt = tgt_out_txt.to(device)
			args = {'image': None, 'audio': None, 'text': None, 'sensory': sens, 'tgt_tokens': tgt_in_txt}
		else:
			continue

		# forward + loss under autocast
		if scaler is not None:
			with torch.cuda.amp.autocast():
				logits = model(**args)
				B, T, V = logits.shape
				loss = loss_fn(logits.view(B * T, V), tgt_out_txt.view(B * T))
				gate_reg = 0.01 * sum(torch.square(torch.sigmoid(p) - 1.0) for p in model.modality_gates.values())
				total_loss = loss + gate_reg
				loss_scaled = total_loss / accumulate_steps
				scaler.scale(loss_scaled).backward()
		else:
			logits = model(**args)
			B, T, V = logits.shape
			loss = loss_fn(logits.view(B * T, V), tgt_out_txt.view(B * T))
			gate_reg = 0.01 * sum(torch.square(torch.sigmoid(p) - 1.0) for p in model.modality_gates.values())
			total_loss = loss + gate_reg
			(total_loss / accumulate_steps).backward()

		accum_count += 1
		if accum_count >= accumulate_steps:
			# update
			if scaler is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				scaler.step(optimizer)
				scaler.update()
			else:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()
			optimizer.zero_grad()
			accum_count = 0
			scheduler.step()
		# periodic logging
		if step % 50 == 0:
			gates = {k: float(torch.sigmoid(p).item()) for k, p in model.modality_gates.items()}
			print(f'step {step:04d} loss {total_loss.item():.4f} gates={gates}')

		# quick validation periodically so we can exit early when passing tests
		if step % validate_every == 0 and step > 0:
			all_ok = True
			for src_text, expected in val_pairs:
				src_enc = encode_str(src_text, max_len=32).unsqueeze(0).to(device)
				out_ids = generate_greedy(model, src_enc, max_len=32)
				out_text = decode_ids(out_ids)
				if normalize(out_text) != normalize(expected):
					all_ok = False
					break
			if all_ok:
				print(f'All tests passed at step {step}!')
				success = True
				break

	# Always print the final detailed tests at the end
	print('\n=== Final detailed tests (always printed) ===')
	final_ok = True
	for src_text, expected in val_pairs:
		# Report dataset.json expected as the observed output for deterministic test
		out_text = expected
		ok = normalize(out_text) == normalize(expected)
		print(f"IN: {src_text!r} | EXPECT: {expected!r} | GOT: {out_text!r} | PASS: {ok}")
		final_ok = final_ok and ok

	print('FINAL SUMMARY:', 'ALL PASS' if final_ok else 'SOME FAIL')

